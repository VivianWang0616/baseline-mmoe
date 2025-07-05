import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from collections import defaultdict
import os
import random



class UnifiedDataset(Dataset):
    def __init__(self, data_tensor):
        """
        Args:
            data_tensor: shape [num_entries, 10, max_label_size]
                        结构说明:
                        [0,:]: user_idx (直接取[0,0]的值)
                        [1,:]: item_idx (直接取[1,0]的值)
                        [2,:]: rating (直接取[2,0]的原始值)
                        [3-9,:]: 7个attributes的one-hot编码
        """
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # 直接取user_idx、item_idx和rating的原始值
        user_idx = entry[0, 0].item()  # 取[0,0]位置的值
        item_idx = entry[1, 0].item()  # 取[1,0]位置的值
        rating = entry[2, 0].item()  # 取[2,0]位置的rating原始值
        # 提取7个attributes的one-hot labels
        attributes = entry[3:10]  # shape [7, max_label_size]

        return (
            torch.tensor([user_idx, item_idx, rating], dtype=torch.float32),  # [3]
            attributes  # [7, max_label_size]
        )


def split_data(data_tensor):
    # 建立用户到数据索引的映射
    user_to_indices = defaultdict(list)
    for idx in range(len(data_tensor)):
        user_idx = data_tensor[idx, 0, 0].item()  # 直接取user_idx的值
        user_to_indices[user_idx].append(idx)

    # 分割数据索引
    train_indices = []
    val_indices = []
    test_indices = []
    count = 0
    for user, indices in user_to_indices.items():
        np.random.shuffle(indices)

        n = len(indices)
        if  n >=10:
            train_end = int(n * 0.7)
            val_end = train_end + int(n * 0.15)
        else: #small sample
            train_end = max(1, int(n * 0.7))
            val_end = train_end + 1
            # test_end = n

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])


    return train_indices, val_indices, test_indices


def load_data(data_tensor, batch_size):
    # 加载数据
    # data_tensor = torch.load(os.path.join(data_dir, "user_item_opinion_one_hot_matrix_4.4.pt"))
    # print(f"加载数据形状: {data_tensor.shape}")  # 调试用

    # 分割数据
    train_indices, val_indices, test_indices = split_data(data_tensor)

    # 创建完整数据集
    full_dataset = UnifiedDataset(data_tensor)

    # 创建基于索引的子集
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    test_subset = torch.utils.data.Subset(full_dataset, test_indices)

    # 自定义collate_fn
    def collate_fn(batch):
        ratings = torch.stack([item[0] for item in batch])  # [batch_size, 3]
        opinions = torch.stack([item[1] for item in batch])  # [batch_size, 7, max_label_size]
        return ratings, opinions

    # 创建DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 验证rating是否正确加载
    sample_ratings, _ = next(iter(train_loader))
    print(
        f"训练集首个batch的rating范围: {sample_ratings[:, 2].min().item():.2f} 到 {sample_ratings[:, 2].max().item():.2f}")

    return train_loader, val_loader, test_loader




#==============generate dataloader for group-item data===============
class GroupOpinionDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def collate_fn(batch):
    # 自动将 batch 组织成 dict of lists/tensors
    return {
        'user_idx': torch.tensor([b['user_idx'] for b in batch], dtype=torch.long),
        'item_idx': torch.tensor([b['item_idx'] for b in batch], dtype=torch.long),
        'rating': torch.tensor([b['rating'] for b in batch], dtype=torch.float),
        'user_opinion_label': torch.stack([b['user_opinion_label'] for b in batch]),  # [B, 7, 4]
        'group_user_ids': [b['group_user_ids'] for b in batch],                       # List[List[int]]
        'group_labels': [b['group_labels'] for b in batch],                           # List[Tensor[G, 7, 4]]
        'group_sum_label': torch.stack([b['group_sum_label'] for b in batch]),
    }
def load_indices(file):
    with open(file, 'r') as f:
        indices = f.read().strip().split(' ')
        return [int(i) for i in indices]

def load_group_data_save_index(data, batch_size, seed=42):
    random.seed(seed)
    # # data = torch.load(path)
    #
    # user_to_items = defaultdict(list)
    # for idx, sample in enumerate(data):
    #     sample['global_idx'] = idx  # 新增: 给每个样本添加其全局 index
    #     u = sample['user_idx']
    #     user_to_items[u].append(sample)
    #
    # train_data, val_data, test_data = [], [], []
    #
    # train_indices, val_indices, test_indices = [], [], []  # 新增: 记录索引
    #
    # for u, samples in user_to_items.items():
    #     random.shuffle(samples)
    #     n = len(samples)
    #     if n == 1:
    #         train_data.append(samples[0])
    #         val_data.append(samples[0])
    #         test_data.append(samples[0])
    #         train_indices.append(str(samples[0]['global_idx']))
    #         # val_indices.append(str(samples[0]['global_idx']))
    #         # test_indices.append(str(samples[0]['global_idx']))
    #     elif n == 2:
    #         train_data.append(samples[0])
    #         val_data.append(samples[1])
    #         test_data.append(samples[1])
    #         train_indices.append(str(samples[0]['global_idx']))
    #         val_indices.append(str(samples[1]['global_idx']))
    #         # test_indices.append(str(samples[1]['global_idx']))
    #     elif n == 3:
    #         train_data.append(samples[0])
    #         val_data.append(samples[1])
    #         test_data.append(samples[2])
    #         train_indices.append(str(samples[0]['global_idx']))
    #         val_indices.append(str(samples[1]['global_idx']))
    #         test_indices.append(str(samples[2]['global_idx']))
    #     elif n > 3 and n < 10:
    #         n_train = max(1, int(n * 0.5)) #0.8/0.15
    #         n_val = int(n * 0.2)
    #         train_data += samples[:n_train]
    #         val_data += samples[n_train:n_val]
    #         test_data += samples[n_val:]
    #
    #         train_indices += [str(s['global_idx']) for s in samples[:n_train]]
    #         val_indices += [str(s['global_idx']) for s in samples[n_train:n_val]]
    #         test_indices += [str(s['global_idx']) for s in samples[n_val:]]
    #     else:
    #         n_train = int(n * 0.5)
    #         n_val = int(n * 0.2)
    #         train_data += samples[:n_train]
    #         val_data += samples[n_train:n_train + n_val]
    #         test_data += samples[n_train + n_val:]
    #
    #         train_indices += [str(s['global_idx']) for s in samples[:n_train]]
    #         val_indices += [str(s['global_idx']) for s in samples[n_train:n_train + n_val]]
    #         test_indices += [str(s['global_idx']) for s in samples[n_train + n_val:]]
    #
    # print(f"📦 Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    #
    # # 保存 index 文件
    # with open('data/data_test/train_indices.index', 'w') as f:
    #     f.write(' '.join(train_indices))
    # with open('data/data_test/val_indices.index', 'w') as f:
    #     f.write(' '.join(val_indices))
    # with open('data/data_test/test_indices.index', 'w') as f:
    #     f.write(' '.join(test_indices))

  #=====================use indices to avoid random split==========================
    full_dataset = GroupOpinionDataset(data)

    # 加载 train, val, test indices
    train_indices = load_indices('data/train_indices.index')
    val_indices = load_indices('data/val_indices.index')
    test_indices = load_indices('data/test_indices.index')

    # 构造 Subset dataset
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#======================================================
    # train_loader = DataLoader(GroupOpinionDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(GroupOpinionDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # test_loader = DataLoader(GroupOpinionDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    return train_loader, val_loader, test_loader

def load_group_data(data, batch_size, seed=42):
    random.seed(seed)
    # data = torch.load(path)

    # 用户 u → 该用户相关的数据项列表
    user_to_items = defaultdict(list)
    for sample in data:
        u = sample['user_idx']
        user_to_items[u].append(sample)

    train_data, val_data, test_data = [], [], []

    for u, samples in user_to_items.items():
        random.shuffle(samples)
        n = len(samples)
        if n == 1:
            train_data.append(samples[0])
            val_data.append(samples[0])
            test_data.append(samples[0])
        elif n == 2:
            train_data.append(samples[0])
            val_data.append(samples[1])
            test_data.append(samples[1])
        elif n == 3:
            train_data.append(samples[0])
            val_data.append(samples[1])
            test_data.append(samples[2])
        elif n > 3 and n < 10:
            n_train = max(1, int(n * 0.6))
            # n_val = n_train + 2
            n_val = int(n * 0.3)
            train_data += samples[:n_train]
            val_data += samples[n_train:n_val]
            test_data += samples[n_val:]
        else:
            n_train = int(n * 0.6)
            n_val = int(n * 0.3)
            train_data += samples[:n_train]
            val_data += samples[n_train:n_train + n_val]
            test_data += samples[n_train + n_val:]


    print(f"📦 Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")

    train_loader = DataLoader(GroupOpinionDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(GroupOpinionDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(GroupOpinionDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
