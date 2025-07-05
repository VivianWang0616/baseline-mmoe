####use same index for baseline as our model


# import pickle
# import random
# from collections import defaultdict
#
# # === 加载原始数据 ===
# with open("/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/reviews.pickle", "rb") as f:
#     reviews = pickle.load(f)  # List[Dict], 每个含有 user_id/item_id
#
# # === 构建 user -> [sample_indices] 映射 ===
# user_to_indices = defaultdict(list)
#
# for idx, entry in enumerate(reviews):
#     user_id = entry.get("user")
#     if user_id is not None:
#         user_to_indices[user_id].append(idx)
#
# # === 划分并收集 train / val / test index ===
# train_indices = []
# val_indices = []
# test_indices = []
#
# for user, indices in user_to_indices.items():
#     if len(indices) < 3:
#         # 如果太少，全部放入 train（或根据需要处理）
#         train_indices.extend(indices)
#         continue
#
#     random.shuffle(indices)
#     n = len(indices)
#     n_train = int(n * 0.3)
#     n_val = int(n * 0.3)
#
#     train_indices.extend(indices[:n_train])
#     val_indices.extend(indices[n_train:n_train + n_val])
#     test_indices.extend(indices[n_train + n_val:])
#
# # === 可选：打乱最终列表（如果用于 batch shuffle）
# random.shuffle(train_indices)
# random.shuffle(val_indices)
# random.shuffle(test_indices)
#
# # === 保存为 index 文件 ===
# def save_indices(filename, indices):
#     with open(filename, "w") as f:
#         f.write(" ".join(str(i) for i in indices))
#
# save_indices("/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/data_index/train.index", train_indices)
# save_indices("/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/data_index/validation.index", val_indices)
# save_indices("/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/data_index/test.index", test_indices)
#
# print(f"✅ 保存完成：train ({len(train_indices)}), val ({len(val_indices)}), test ({len(test_indices)})")


#=======================与mmoe模型数据保持一致==========
import torch
import json
import pickle

# === 加载所需文件 ===

# 加载 pt 文件
pt_data = torch.load('/home/hongren/review_gen_MMOE/data/Yelp_mmoe/yelp_with_group_data_6.24_filtered.pt')  # 替换为你的实际 pt 文件路径

# 加载 user2idx 与 item2idx
with open('/home/hongren/review_gen_MMOE/new_processed_data/user2idx.json', 'r') as f:
    user2idx = json.load(f)

with open('/home/hongren/review_gen_MMOE/new_processed_data/item2idx.json', 'r') as f:
    item2idx = json.load(f)

# 生成 idx2user 与 idx2item 反向映射
idx2user = {v: k for k, v in user2idx.items()}
idx2item = {v: k for k, v in item2idx.items()}

# 加载 reviews.pickle
with open('/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/reviews.pickle', 'rb') as f:
    reviews = pickle.load(f)

# === 读取 index 文件 ===

def load_indices(file):
    with open(file, 'r') as f:
        indices = f.read().strip().split(' ')
        return [int(i) for i in indices]

train_indices = load_indices('/home/hongren/review_gen_MMOE/data/train_indices.index')
val_indices = load_indices('/home/hongren/review_gen_MMOE/data/val_indices.index')
test_indices = load_indices('/home/hongren/review_gen_MMOE/data/test_indices.index')

# === 根据索引映射回 user_id, item_id ===

def get_user_item_pairs(indices, pt_data, idx2user, idx2item):
    pairs = []
    for idx in indices:
        sample = pt_data[idx]
        user_idx = sample['user_idx']
        item_idx = sample['item_idx']
        user_id = idx2user[user_idx]
        item_id = idx2item[item_idx]
        pairs.append( (user_id, item_id) )
    return pairs

train_pairs = get_user_item_pairs(train_indices, pt_data, idx2user, idx2item)
val_pairs = get_user_item_pairs(val_indices, pt_data, idx2user, idx2item)
test_pairs = get_user_item_pairs(test_indices, pt_data, idx2user, idx2item)

# === 在 reviews.pickle 中根据 user-item pair 找到 index ===

def find_review_indices(pairs, reviews):
    review_indices = []
    for i, review in enumerate(reviews):
        uid = review['user']
        iid = review['item']
        if (uid, iid) in pairs:
            review_indices.append(str(i))
    return review_indices

# 生成各自的 review indices
train_review_indices = find_review_indices(set(train_pairs), reviews)
val_review_indices = find_review_indices(set(val_pairs), reviews)
test_review_indices = find_review_indices(set(test_pairs), reviews)

# === 保存到文件 ===

with open('/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/data_index/train.index', 'w') as f:
    f.write(' '.join(train_review_indices))
with open('/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/data_index/val.index', 'w') as f:
    f.write(' '.join(val_review_indices))
with open('/home/hongren/review_gen_MMOE/data/baseline_data/baseline_yelp/data_index/test.index', 'w') as f:
    f.write(' '.join(test_review_indices))

print("✅ Generated train.index, val.index, test.index based on reviews.pickle mapping.")
