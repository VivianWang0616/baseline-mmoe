import os
import math
import torch
import argparse
import torch.nn as nn
from transformers import GPT2Tokenizer
from torch.optim import AdamW
from module import RecReg
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')
parser.add_argument('--data_path', type=str, default='/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/reviews.pickle',
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default='/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/data_index/',
                    help='load indexes')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./yelp_mlp/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--rating_reg', type=float, default=0.01,
                    help='regularization on recommendation task')
parser.add_argument('--text_reg', type=float, default=1.0,
                    help='regularization on text generation task')
parser.add_argument('--use_mf', action='store_true',
                    help='otherwise MLP')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
corpus = DataLoader(args.data_path, args.index_dir, tokenizer, args.words)
feature_set = corpus.feature_set
train_data = Batchify(corpus.train, tokenizer, bos, eos, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, tokenizer, bos, eos, args.batch_size)
test_data = Batchify(corpus.test, tokenizer, bos, eos, args.batch_size)

###############################################################################
# Build the model
###############################################################################

nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntoken = len(tokenizer)
model = RecReg.from_pretrained('gpt2', nuser, nitem, args.use_mf)
model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table
model.to(device)
rating_criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=args.lr)

###############################################################################
# Training code
###############################################################################


def train(data):
    # Turn on training mode which enables dropout.
    model.train()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    while True:
        user, item, rating, seq, mask = data.next_batch()  # data.step += 1
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        seq = seq.to(device)  # (batch_size, seq_len)
        mask = mask.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        outputs, rating_p = model(user, item, seq, mask)
        t_loss = outputs.loss
        r_loss = rating_criterion(rating_p, rating)
        loss = args.text_reg * t_loss + args.rating_reg * r_loss
        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_t_loss), cur_r_loss, data.step, data.total_step))
            text_loss = 0.
            rating_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break


def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, seq, mask = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.to(device)  # (batch_size, seq_len)
            mask = mask.to(device)
            outputs, rating_p = model(user, item, seq, mask)
            t_loss = outputs.loss
            r_loss = rating_criterion(rating_p, rating)

            batch_size = user.size(0)
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample, rating_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, rating, seq, _ = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            text = seq[:, :1].to(device)  # bos, (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                if idx == 0:
                    outputs, rating_p = model(user, item, text, None)
                    rating_predict.extend(rating_p.tolist())
                else:
                    outputs, _ = model(user, item, text, None, False)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict, rating_predict


#++==================rating distribution==========import torch

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_rating_distribution(ratings, pred_ratings, save_path="rating_distribution.png"):
    """
    绘制真实评分和预测评分的分布对比图
    修改说明：将pred_rating先进行近似成整数再绘图
    近似规则：
        0.5-1.4 → 1
        1.5-2.4 → 2
        2.5-3.4 → 3
        3.5-4.4 → 4
        4.5-5.4 → 5
        5.5-6.4 → 6
        6.5-7.0 → 7
    """

    # 将预测评分近似为整数
    def round_rating(x):
        if x < 0.5:  # 处理小于0.5的特殊情况
            return 0
        elif x < -0.5:
            return -1
        return int(np.floor(x + 0.5))

    rounded_pred = np.array([round_rating(r) for r in pred_ratings])

    plt.figure(figsize=(12, 6))

    # 真实评分分布
    plt.subplot(1, 2, 1)
    sns.histplot(ratings, bins=range(0, 8), kde=False, color='blue', discrete=True)
    plt.title('Real Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(range(0, 8))  # 显示所有可能的评分值

    # 近似后的预测评分分布
    plt.subplot(1, 2, 2)
    sns.histplot(rounded_pred, bins=range(0, 8), kde=False, color='orange', discrete=True)
    plt.title('Rounded Predicted Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(range(0, 8))  # 显示所有可能的评分值

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"评分分布图已保存至 {save_path}")

# def plot_rating_distribution(ratings, pred_ratings, output_dir="results"):
#     """
#     绘制真实评分和预测评分的分布对比图
#     """
#     plt.figure(figsize=(12, 6))
#
#     # 真实评分分布
#     plt.subplot(1, 2, 1)
#     sns.histplot(ratings, bins=10, kde=True, color='blue')
#     plt.title('Real Rating Distribution')
#     plt.xlabel('Rating')
#     plt.ylabel('Count')
#
#     # 预测评分分布
#     plt.subplot(1, 2, 2)
#     sns.histplot(pred_ratings, bins=10, kde=True, color='orange')
#     plt.title('Predicted Rating Distribution')
#     plt.xlabel('Rating')
#     plt.ylabel('Count')
#
#     plt.tight_layout()
#     plt.savefig(f"rating_distribution.png")
#     plt.close()

import json
def generate_jsonl(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    results = []  # 存储所有结果的列表
    idss_predict = []  # 保留原始ID列表输出

    all_ratings = []
    all_pred_ratings = []

    with torch.no_grad():
        while True:
            # 获取当前batch的数据
            user, item, rating, seq, _ = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            text = seq[:, :1].to(device)  # bos, (batch_size, 1)

            # 生成文本
            for idx in range(seq.size(1)):
                if idx == 0:
                    outputs, rating_p = model(user, item, text, None)
                    # rating_predict.extend(rating_p.tolist())
                else:
                    outputs, _ = model(user, item, text, None, False)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                # outputs = model(user, item, text, None)
                # last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                # word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
                text = torch.cat([text, token], 1)  # (batch_size, len++)


            # 获取生成的token IDs（移除BOS）
            ids = text[:, 1:].tolist()  # (batch_size, seq_len)
            idss_predict.extend(ids)

            # 收集当前batch的原始信息
            batch_users = user.cpu().tolist()
            batch_items = item.cpu().tolist()
            batch_ratings = rating.cpu().tolist()
            batch_pred_ratings = rating_p.cpu().tolist()  # 假设模型输出预测评分

            # all_users.extend(batch_users)
            # all_items.extend(batch_items)
            all_ratings.extend(batch_ratings)
            all_pred_ratings.extend(batch_pred_ratings)

            # 临时存储当前batch的结果（先只存ID，最后统一转换文本）
            for u, i, r, id_list in zip(batch_users, batch_items, batch_ratings, ids):
                results.append({
                    "user_id": u,
                    "item_id": i,
                    "rating": r,
                    "token_ids": id_list  # 先存储token IDs
                })

            if data.step == data.total_step:
                break

    # 统一转换token IDs为文本（保持与原文一致的方式）
    tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predict]
    text_predict = [' '.join([token for token in tokens if token != '<eos>'])
                    for tokens in tokens_predict]
    # text_predict = [' '.join(tokens) for tokens in tokens_predict]
    # tokens_test = [ids2tokens(ids[1:], tokenizer, eos) for ids in data.seq.tolist()]
    # def clean_eos(tokens):
    #     # 找到第一个<eos>的位置
    #     eos_pos = len(tokens)
    #     if '<eos>' in tokens:
    #         eos_pos = tokens.index('<eos>') + 1  # 保留第一个<eos>之前的内容
    #     cleaned = tokens[:eos_pos]
    #     # 移除所有剩余的<eos>
    #     cleaned = [token for token in cleaned if token != '<eos>']
    #     return ' '.join(cleaned)
    #
    # text_predict = [clean_eos(tokens) for tokens in tokens_predict]

    # 更新results中的文本内容
    plot_rating_distribution(all_ratings, all_pred_ratings)
    for i, res in enumerate(results):
        res["generated_text"] = text_predict[i]
        del res["token_ids"]  # 移除临时的token IDs字段

    # 写入JSONL文件
    # output_path = os.path.join(args.checkpoint, 'pepler_predict_text.jsonl')
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     for record in results:
    #         f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 保持与原始函数相同的返回值
    return idss_predict

# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_t_loss, val_r_loss = evaluate(val_data)
    val_loss = val_t_loss + val_r_loss
    print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
        math.exp(val_t_loss), val_r_loss, val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    # model = torch.load(f).to(device)
    model = torch.load(f, weights_only=False).to(device)


# Run on test data.
test_t_loss, test_r_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} on test | End of training'.format(math.exp(test_t_loss), test_r_loss))
print(now_time() + 'Generating text')
idss_predicted, rating_predicted = generate(test_data)
idss_predicted_new = generate_jsonl(test_data)
# rating
predicted_rating = [(r, p) for (r, p) in zip(test_data.rating.tolist(), rating_predicted)]
RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'MAE {:7.4f}'.format(MAE))
# text
# tokens_test = [ids2tokens(ids[1:], tokenizer, eos) for ids in test_data.seq.tolist()]
# tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
# BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
# print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
# BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
# print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
# USR, USN = unique_sentence_percent(tokens_predict)
# print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
# feature_batch = feature_detect(tokens_predict, feature_set)
# DIV = feature_diversity(feature_batch)  # time-consuming
# print(now_time() + 'DIV {:7.4f}'.format(DIV))
# FCR = feature_coverage_ratio(feature_batch, feature_set)
# print(now_time() + 'FCR {:7.4f}'.format(FCR))
# FMR = feature_matching_ratio(feature_batch, test_data.feature)
# print(now_time() + 'FMR {:7.4f}'.format(FMR))
# text_test = [' '.join(tokens) for tokens in tokens_test]
# text_predict = [' '.join(tokens) for tokens in tokens_predict]
# ROUGE = rouge_score(text_test, text_predict)  # a dictionary
# for (k, v) in ROUGE.items():
#     print(now_time() + '{} {:7.4f}'.format(k, v))
# text_out = ''
# for (real, fake) in zip(text_test, text_predict):
#     text_out += '{}\n{}\n\n'.format(real, fake)
# with open(prediction_path, 'w', encoding='utf-8') as f:
#     f.write(text_out)
# print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
