import argparse
import os
import json
import re
import random
import numpy as np
# from googletrans import Translator
import torch
from collections import defaultdict
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# import torch.nn as nn
# from data_process import load_data, load_friends_opinions, convert_labels_to_one_hot, gen_true_ratings_similar_users
# from models.transformer_mmoe import MultiTaskModel
# from models.group_mmoe import MultiTaskModel
from models.diff_att_mmoe import MultiTaskModel
# from models.base_mmoe import MultiTaskModel

# from utils import compute_loss, load_opinions, save_best_model
# from process_matrix2 import OpinionSimilarityProcessor, EmbeddingModel
# import matplotlib.pyplot as plt
from datetime import datetime
import logging
import warnings
from data_process import prepare_dataloaders, split_user_data

from torch.utils.data import DataLoader, TensorDataset, random_split
# from mmoe_tr_eval import MMOE_trainer
from group_mmoe_tr_eval import group_MMOE_trainer
# from base_mmoe_tr_eval import group_MMOE_trainer
from data.data_prepare import load_data, load_group_data, load_group_data_save_index

# from peter_dataloader import DataLoader, Batchify

# Initialize logging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
log_filename = f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Optional: Print logs to console as well
logging.getLogger().addHandler(logging.StreamHandler())





def parse_args():
    # attribute_files = '{"food": "/home/hongren/review_gen_MMOE/data/food_clusters.json", "service": "/home/hongren/review_gen_MMOE/data/service_clusters.json", "ambience": "/home/hongren/review_gen_MMOE/data/ambience_clusters.json", "price": "/home/hongren/review_gen_MMOE/data/price_clusters.json", "location": "/home/hongren/review_gen_MMOE/data/location_clusters.json", "cleanliness": "/home/hongren/review_gen_MMOE/data/cleanliness_clusters.json", "parking": "/home/hongren/review_gen_MMOE/data/parking_clusters.json"}'
    parser = argparse.ArgumentParser(description="Multitask Model for Rating and Review Generation")
    base_path = "/home/hongren/review_gen_MMOE/new_processed_data/"
    # base_path = "/home/hongren/hongren_mmoe/data/"

    # user_file = os.path.join(base_path, "filtered_yelp_users.json")
    # business_file = os.path.join(base_path, "filtered_yelp_businesses.json")
    # rating_file = os.path.join(base_path, "reshaped_rating_matrix.pt") # Shape: [num_users, 3] -> (user_id, item_id, rating)

    # opinion_file = os.path.join(base_path, "reshaped_opinion_matrix.pt") # Shape: [num_users, max_items_per_user, 8(7+item_idx), max_label_size(176)]



    # data_file = os.path.join(base_path, "user_item_opinion_one_hot_matrix_4.4.pt")
    # previous_data = torch.load(data_file)

    # friends_opinions_files = os.path.join(base_path, "friends_opinions.json")
    # friends_items_pairs_file = os.path.join(base_path, "merged_friends_item_pairs.json")
    # one_hot_label_matrix = os.path.join(base_path, "user_item_one_hot_labels_matrix.pt")
    # review_file = os.path.join(base_path, "filtered_yelp_reviews.json")
    # label_file = os.path.join(base_path, "gpt_sec_cleaned_standardized_labels.json")
    parser.add_argument('--base_path', type=str, default=base_path, help="Base path to all data files")

    parser.add_argument('--data_path', type=str, default='/home/hongren/review_gen_MMOE/Yelp/reviews.pickle',
                        help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, default='/home/hongren/review_gen_MMOE/Yelp/1/',
                        help='load indexes')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='keep the most frequent words in the dict')
    parser.add_argument('--words', type=int, default=15,
                        help='number of words to generate for each sample')

    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument('--emb_size', type=int, default=64, help="Embedding size")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--num_experts', type=int, default=3, help="Number of experts")
    parser.add_argument('--num_attributes', type=int, default=7, help="Number of attributes")
    parser.add_argument('--sentiment_loss', type=int, default=1, help="set value more than 0 to use this loss")
    parser.add_argument('--use_cuda', action='store_true', help="Use CUDA if available")
    parser.add_argument("--model_name", type=str, default="MMOE",
                        help="Choose a model: MMOE, BaselineA, BaselineB")
    parser.add_argument("--save_dir", type=str, default="./saved_models/MMOE_3.21",
                        help="Directory to save best model checkpoints")

    # parser.add_argument('--gpu', type=int, default=1, help="GPU id to use")
    args = parser.parse_args()

    # Attach file paths to args
    # args.user_file = user_file
    # args.business_file = business_file
    # args.rating_file = rating_file
    # args.opinion_file = opinion_file
    # args.label_file = label_file
    # args.friends_opinions_files = friends_opinions_files
    # args.friends_items_pairs_file = friends_items_pairs_file
    # args.one_hot_label_matrix = one_hot_label_matrix
    # args.data_file = data_file
    # args.review_file = review_file

    return args




def main():
    warnings.filterwarnings("ignore")
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    print("Loading rating and opinions matrix...")
    # opinions = load_opinions(args.opinion_file)
    # opinion_data = torch.load(args.opinion_file)
    # rating_data = torch.load(args.rating_file)
    #=========test model code======================
    # part_group_data = torch.load("/home/hongren/review_gen_MMOE/data/processed_group_opinions_6.11.pt")
    # part_group_data = torch.load("/home/hongren/review_gen_MMOE/data/processed_group_opinions_with_counts_6.17.pt")
    part_group_data = torch.load("/home/hongren/review_gen_MMOE/data/Yelp_mmoe/yelp_with_group_data_6.24_filtered.pt")


    # user_idx = data_part_api[:, 0, 0]
    # data_part_api = torch.load("/home/hongren/review_gen_MMOE/data/processed_opinions_sentiment_4.26.pt")
    # user_idx = data_part_api[:, 0, 0]
    # item_idx = data_part_api[:, 1, 0]
    #
    # # 统计唯一值数量
    # num_users = len(torch.unique(user_idx))
    # num_items = len(torch.unique(item_idx))
    # data = torch.load(args.data_file)
    #==================load peter data===============

    # corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
    # word2idx = corpus.word_dict.word2idx
    # idx2word = corpus.word_dict.idx2word
    # feature_set = corpus.feature_set  # null

    # rating_data_old = torch.load("./new_processed_data/rating_dataset.pt")
    # friends_opinions = load_friends_opinions(args.friends_opinions_files)

    # Load item2idx mapping
    print("Load user_item idx...")
    with open("./new_processed_data/item2idx.json", "r") as f:
        item2idx = json.load(f)
    with open("./new_processed_data/user2idx.json", "r") as f:
        user2idx = json.load(f)
    #
    # print("Load friends_item pairs matrix...")
    # with open(args.friends_items_pairs_file, "r") as f:
    #     friends_items_pairs = json.load(f)

#===============================user-friends-matrix=========================
    # num_users = len(user2idx)
    # user_idx = sorted(user2idx.values())
    # # max_num_items = max(len(items) for items in user_item_dict.values())
    # max_num_friends = 0  # Will update during processing
    #
    # num_items_per_user = []
    # for user in user_idx:
    #     user_indices = (rating_data_old[:, 0] == user)
    #     items = rating_data_old[user_indices, 1]
    #     # ratings = rating_data_old[user_indices, 2]
    #     num_items_per_user.append(items.shape[0])
    #
    # # Initialize container
    # max_num_items = rating_data.shape[1]
    # user_items = defaultdict(list)
    # for row in rating_data:
    #     user_idx = row[0][0].item()
    #     item_list = row[:,1].tolist()  # Only use first two elements
    #
    #     user_items[user_idx].append(item_list)
    #
    # # max_num_items = max(len(v) for v in user_items.values())
    #
    # # Step 2: Build user-item-friends structure
    # friend_matrix = [[[] for _ in range(max_num_items)] for _ in range(num_users)]
    # max_num_friends = 0
    #
    # # Build reverse mappings
    # idx2user = {v: k for k, v in user2idx.items()}
    # idx2item = {v: k for k, v in item2idx.items()}
    #
    # for user_idx, item_list in user_items.items():
    #     orig_user_id = idx2user[user_idx]
    #     for i, item_idx in enumerate(item_list[0]):
    #         if i < num_items_per_user[int(user_idx)]:
    #             orig_item_id = idx2item[item_idx]
    #
    #             friend_ids = []
    #             if orig_user_id not in friends_items_pairs.keys():
    #                 print(f"⚠️ User {orig_user_id} not found in friends_items_pairs.")
    #             for entry in friends_items_pairs.get(orig_user_id, []):
    #                 if entry["item_id"] == orig_item_id and entry["friend_id"] in user2idx:
    #                     friend_ids.append(user2idx[entry["friend_id"]])
    #
    #             friend_matrix[user_idx][i] = friend_ids
    #             max_num_friends = max(max_num_friends, len(friend_ids))
    #         else:
    #             break
    #
    # # Step 3: Pad and convert to tensor
    # padded_tensor = torch.full((num_users, max_num_items, max_num_friends), fill_value=-1, dtype=torch.long)
    #
    # for u in range(num_users):
    #     for i in range(len(friend_matrix[u])):
    #         f_list = friend_matrix[u][i]
    #         if f_list:
    #             padded_tensor[u, i, :len(f_list)] = torch.tensor(f_list[:max_num_friends])
    #
    # # Save
    # torch.save(padded_tensor, "user_item_friend_matrix.pt")
    # print(f"✅ user_item_friend_matrix.pt saved with shape: {padded_tensor.shape}")










#======================================================================





    # print("Load one hot labels matrix for opinions...")
    # one_hot_label_data = torch.load(args.one_hot_label_matrix)
    # attributes = ['Ambience', 'Cleanliness', 'Food', 'Location', 'Parking', 'Price', 'Service']

    # Prepare DataLoaders
    print("Prepare Dataloaders...")

    train_loader, val_loader, test_loader = load_group_data_save_index(part_group_data, batch_size=args.batch_size)

    num_users = len(user2idx.values()) ## embedding不会出错
    num_items = len(item2idx.values())


    if args.model_name == "MMOE":
        print("Initializing MMOE model...")
        mmoe_model = MultiTaskModel(
            num_users,
            num_items,
            input_dim=args.emb_size * 7, #448
            # emb_dim=args.emb_size,
            expert_dim=64,
            num_experts=args.num_experts,
            sentiment_loss=args.sentiment_loss,
            num_attributes=args.num_attributes,
            # max_label_size=max_label_size
        ).to(device)

    # Train Model
    mmoe_optimizer = torch.optim.AdamW(mmoe_model.parameters(), lr=0.00001, weight_decay=1e-4)
    # mmoe_optimizer = torch.optim.Adam(mmoe_model.parameters(), lr=0.0003)
    # mmmoe_scheduler = torch.optim.lr_scheduler.StepLR(mmoe_optimizer, step_size=5, gamma=0.25)
    # mmoe_optimizer = torch.optim.SGD(mmoe_model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(mmoe_optimizer, 5, gamma=0.25)

    # === Trainer Instance ===
    # user-items for rating prediction; user-friends-items for opinions prediction
    # trainer = MMOE_trainer(mmoe_model, train_loader, val_loader, test_loader, mmoe_optimizer, scheduler, device, args.save_dir)
    ##=======group data training
    trainer = group_MMOE_trainer(mmoe_model, train_loader, val_loader, test_loader, mmoe_optimizer, scheduler, device, args.save_dir)

    # === Train & Evaluate Model ===
    num_epochs = 50
    # trainer.train(num_epochs, is_predicted_opinion = True)   # original:500
    #
    # === Final Testing ===
    trainer.test(is_predicted_opinion=True)








if __name__ == "__main__":
    main()
