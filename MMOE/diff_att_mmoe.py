import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional
from torch import Tensor

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        output = src
        attns = []
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)
        attns = torch.stack(attns)
        if self.norm is not None:
            output = self.norm(output)
        return output, attns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# import torch
# import torch.nn as nn
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MLP_as_MMOE(nn.Module):
    def __init__(self, emsize=512, hidden_size=512, num_layers=2):
        super(MLP_as_MMOE, self).__init__()
        self.first_layer = nn.Linear(emsize, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, hidden):  # hidden: (batch_size, emsize)
        out = self.sigmoid(self.first_layer(hidden))  # -> (batch_size, hidden_size)
        for layer in self.layers:
            out = self.sigmoid(layer(out))            # -> (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(out))  # -> (batch_size,)
        return rating


class MMOE_MLP(nn.Module):
    def __init__(self, emsize=512):
        super(MMOE_MLP, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):
        mlp_vector = self.sigmoid(self.linear1(hidden))
        # rating = self.sigmoid(self.linear2(mlp_vector))
        rating = torch.squeeze(self.linear2(mlp_vector))
        # rating = rating.squeeze(-1)
        return rating

# import torch
# import torch.nn as nn
# import copy
#
# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MMOE_predictor_new(nn.Module):
    def __init__(self, emsize=512, hidden_size=512, num_layers=2):
        super(MMOE_predictor_new, self).__init__()
        self.first_layer = nn.Linear(emsize, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(hidden_layer, num_layers)

        self.gelu = nn.GELU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, hidden):
        x = self.gelu(self.first_layer(hidden))  # (batch_size, hidden_size)
        for layer in self.layers:
            x = self.gelu(layer(x))              # (batch_size, hidden_size)
        score = self.last_layer(x).squeeze(-1)   # (batch_size,)
        return score


class MMOE_predictor(nn.Module):
    def __init__(self, emsize=512):
        super(MMOE_predictor, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):
        mlp_vector = self.gelu(self.linear1(hidden))
        # rating = self.sigmoid(self.linear2(mlp_vector))
        score = self.linear2(mlp_vector).squeeze(-1)
        # score = torch.squeeze(self.linear2(mlp_vector))
        # rating = rating.squeeze(-1)
        return score

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, expert_dim, nhead=4, dropout_rate=0.1): #########0.1
        super(ExpertNetwork, self).__init__()
        # self.layer = nn.Linear(input_dim, expert_dim)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            # nn.ReLU(),
            nn.GELU(),
            # nn.BatchNorm1d(expert_dim),
            nn.LayerNorm(expert_dim),
            nn.Dropout(dropout_rate)
        )
        nn.init.xavier_uniform_(self.layer[0].weight)

        # Transformer encoder layer
        self.transformer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=expert_dim*4,
            dropout=0.5    ####dropout=0.5
        )

        # Final MLP for expert output
        # self.mlp = MLP(emsize=expert_dim)

    def forward(self, x):
        self.layer(x)
        # Add sequence dimension for transformer (sequence length = 1)
        x = x.unsqueeze(0)  # [1, batch_size, expert_dim]

        # Process through transformer
        x, _ = self.transformer(x)  # [1, batch_size, expert_dim]

        # Remove sequence dimension
        x = x.squeeze(0)  # [batch_size, expert_dim]

        # Final MLP processing
        # x = self.mlp(x)  # [batch_size, 1]
        # x = x.squeeze(-1)  # [batch_size]

        return x

class GateNetwork(nn.Module):
    def __init__(self, num_experts, input_dim):
        super(GateNetwork, self).__init__()
        self.gate_layer = nn.Linear(input_dim, num_experts)
        # self.temperature = temperature
        nn.init.xavier_uniform_(self.gate_layer.weight)

    def forward(self, x):
        # Generate gating weights using softmax
        # return F.softmax(self.gate_layer(x), dim=1)

        return F.softmax(self.gate_layer(x), dim=1)


class ReviewTransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_attributes, num_labels=4):
        super(ReviewTransformerEncoder, self).__init__()
        self.num_attributes = num_attributes

        # PETER风格的Transformer编码层
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

        # 情感注意力模块
        # self.sentiment_attention = nn.Sequential(
        #     nn.Linear(input_dim + num_labels, 64),  # 融合特征和标签
        #     nn.GELU(),
        #     nn.Linear(64, 1)
        # )

        # 标签embedding层 (4种标签)
        self.label_embedding = nn.Embedding(5, input_dim) #4种label加一种无效label

        # 用于融合标签信息的门控机制
        self.label_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )

        # self.max_label_size = max_label_size
        #
        # self.norm = nn.LayerNorm(input_dim)
        # self.scale = math.sqrt(input_dim)

    def forward(self, x, true_labels=None):
        # x: [batch_size, num_attributes, input_dim] (预测的review特征)
        # true_labels: [batch_size, num_attributes] (真实的label索引)

        batch_size = x.size(0)




        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # attn_weights = F.softmax(attn_scores, dim=-1)

        # 2. 如果有真实标签，则融入标签信息
        # === 第一阶段：标签融合 ===
        if true_labels is not None and self.training:
            # 1. 将无效标签（-1）映射为特殊索引4
            adjusted_labels = true_labels.clone()
            adjusted_labels[true_labels == -1] = 4  # 使用额外的embedding

            # 2. 获取标签embedding

            label_emb = self.label_embedding(adjusted_labels)  # [batch, num_attrs, input_dim]
            # print("Any NaN in label_emb:", torch.isnan(label_emb).any().item())

            # 3. 门控融合
            gate_input = torch.cat([x, label_emb], dim=-1)
            gate = self.label_gate(gate_input)
            x = gate * x + (1 - gate) * label_emb #[512]
        #===========transformer======
        # 生成padding mask（无效标签位置为True）
        # print("Any NaN in unmask gate:", torch.isnan(x).any().item())
        if true_labels is not None:
            padding_mask = (true_labels == -1)  # [batch, num_attrs]
        else:
            padding_mask = None
        x = x.transpose(0, 1)
        x, _ = self.transformer(x, src_key_padding_mask=padding_mask)
        x = x.transpose(0, 1)  # 恢复维度 #[512,7,448]

        # print("Any NaN in unmask output:", torch.isnan(unmask_output).any().item())

        if true_labels is not None:
            valid_mask = (true_labels != -1).unsqueeze(-1).float()  # [batch, num_attrs, 1]
            weighted = x * valid_mask  # 无效位置置零

            # 安全均值计算
            valid_counts = valid_mask.sum(dim=1)  # [batch, 1]
            output = torch.where(
                valid_counts > 0,
                weighted.sum(dim=1) / valid_counts,  # 有效样本：加权平均
                torch.zeros_like(weighted.sum(dim=1))  # 全无效样本：输出零向量
            )
        else:
            output = x.mean(dim=1)  # 无mask时直接平均 # 仅有效位置均值



        # attn_scores = x.mean(dim=1) #[512,448]
        # # 1. 标准self-attention
        # q = self.query(x)  # [batch_size, num_attributes, input_dim]
        # k = self.key(x)
        # v = self.value(x)
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch, num_attrs, num_attrs]

        # 3. 可选：Mask无效标签的注意力（若需要）
        # if true_labels is not None:
        #     invalid_mask = (true_labels == -1).unsqueeze(1)  # [batch, 1, num_attrs]
        #     attn_scores = attn_scores.masked_fill(invalid_mask, float('-inf'))
        #
        # attn_weights = F.softmax(attn_scores, dim=-1)

        # 4. 应用注意力
        # weighted = torch.matmul(attn_weights, v)  # [batch, num_attrs, input_dim]
        # === 第三阶段：池化输出 ===
        # 1. 计算有效attr的均值（排除被mask的位置）
        # if true_labels is not None:
        #     valid_mask = (true_labels != -1).unsqueeze(-1)  # [batch, num_attrs, 1]
        #     weighted = weighted * valid_mask  # 置零无效位置
        #     output = weighted.sum(dim=1) / (valid_mask.sum(dim=1) + 1e-6)  # [batch, input_dim]
        # else:
        #     output = weighted.mean(dim=1)  # 无mask时直接平均
        #

        return output

        # if true_labels is not None:
        #
        #     # 获取标签embedding [batch_size, num_attributes, input_dim]
        #     label_emb = self.label_embedding(true_labels)
        #
        #     # 计算标签相关性权重
        #     label_weights = torch.matmul(q, label_emb.transpose(-2, -1)) / self.scale
        #     label_weights = F.softmax(label_weights, dim=-1)
        #
        #     # 融合预测attention和标签attention
        #     fused_weights = 0.7 * attn_weights + 0.3 * label_weights
        #
        #     # 应用门控机制调整特征
        #     gate_input = torch.cat([x, label_emb], dim=-1)
        #     gate = self.label_gate(gate_input) #[batch, 7, 448]
        #     v = gate * v + (1 - gate) * label_emb
        # else:
        #     fused_weights = attn_weights
        #
        # # 应用attention
        # weighted = torch.matmul(fused_weights, v)
        # return weighted.mean(dim=1)  # [batch_size, input_dim]
#
#
class ReviewAttentionNetwork(nn.Module):
    def __init__(self, input_dim, num_attributes, hidden_dim=64):
        super(ReviewAttentionNetwork, self).__init__()
        self.num_attributes = num_attributes
        self.query = nn.Linear(input_dim, input_dim)
        #============diff to be query============
        self.diff_query = nn.Linear(1, input_dim)
        self.diff_query_proj = nn.Sequential(
            nn.Linear(1, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim)
        )
        # Label-based queries (2 heads)
        num_labels = 4
        self.label_query_proj1 = nn.Linear(num_labels, input_dim)
        self.label_query_proj2 = nn.Linear(num_labels, input_dim)
        #============================================
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = math.sqrt(input_dim)


        initial_sentiment_map = torch.tensor([1.0, 0.7, -0.7, -1.0]).unsqueeze(1)  # [4,1]
        self.sentiment_embedding = nn.Parameter(initial_sentiment_map * torch.randn(4, input_dim))

        self.fuse_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        self.review_score_predictor = MMOE_predictor(input_dim)
        # self.review_score_predictor = MMOE_predictor_new(input_dim)

    def gt_review_score(self, true_labels, rating_feature, mask=None):
        probs = true_labels.float()
        batch_size = probs.size(0)
        sentiment_vecs = torch.matmul(probs, self.sentiment_embedding)  # [B, 7, D]

        if mask is not None:
            sentiment_vecs = sentiment_vecs.masked_fill(mask.unsqueeze(-1), 0.0)
        q = self.query(sentiment_vecs)
        k = self.key(sentiment_vecs)
        v = self.value(sentiment_vecs)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        weighted = torch.matmul(attn_weights, v)  # [batch_size, num_attributes, input_dim]



        output = weighted + sentiment_vecs  # residual

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_counts = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        else:
            valid_counts = torch.tensor(self.num_attributes, dtype=torch.float32, device=probs.device).unsqueeze(
                0).repeat(batch_size, 1)

        pooled = output.sum(dim=1) / valid_counts  # [B, D]

        gate_input = torch.cat([rating_feature, pooled], dim=-1)
        gate = self.fuse_gate(gate_input)  # [B, D]
        final_rep = gate * rating_feature + (1 - gate) * pooled

        review_score = self.review_score_predictor(final_rep).squeeze(-1)  # [B]
        return review_score

    def user_forward(self, opinion_probs, rating_feature, mask=None):
        sentiment_vecs = torch.matmul(opinion_probs, self.sentiment_embedding)  # [B, 7, 448]
        # sentiment_vecs_d = torch.matmul(opinion_probs_d, self.sentiment_embedding)
        # sentiment_vecs = torch.cat([opinion_features, sentiment_vecs], dim=-1)  # [B, 7, 2D]

        if mask is not None:  ##do we need two masks??
            sentiment_vecs = sentiment_vecs.masked_fill(mask.unsqueeze(-1), 0.0)
            # sentiment_vecs_d = sentiment_vecs_d.masked_fill(mask.unsqueeze(-1), 0.0)
        # q = self.diff_query_proj(rating_diff.unsqueeze(-1)).unsqueeze(1)  # [B, 1, D]

        q = self.query(sentiment_vecs)
        k = self.key(sentiment_vecs)
        v = self.value(sentiment_vecs)

        # k_d = self.key(sentiment_vecs_d)
        # v_d = self.value(sentiment_vecs_d)

        # diff = rating_diff.unsqueeze(-1)  # [B, 1]
        # q = self.diff_query(diff).unsqueeze(1)  # [B, 1, D]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), -1e9)

        # Calculate attention scores
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_attributes, num_attributes]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, 7, 7]

        # Apply attention to values
        weighted = torch.matmul(attn_weights, v)  # [batch_size, num_attributes, input_dim]

        output = weighted + sentiment_vecs  # residual
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_counts = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        else:
            valid_counts = torch.tensor(self.num_attributes, dtype=torch.float32,
                                        device=opinion_probs.device).unsqueeze(0).repeat(batch_size, 1)

        pooled = output.sum(dim=1) / valid_counts  # [B, D]
        # output = weighted.squeeze(1) #[B,D]

        # ==========multi-heads multi queries=============
        # === Head 2: label head 1 ===
        # q2 = self.label_query_proj1(opinion_probs)  # [B, 7, D]
        # attn2 = torch.matmul(q2, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]
        # if mask is not None:
        #     attn2 = attn2.masked_fill(mask.unsqueeze(1), -1e9)
        # w2 = F.softmax(attn2, dim=-1)
        # out2 = torch.matmul(w2, v).mean(dim=1)  # [B, D]
        #
        # # === Head 3: label head 2 ===
        # q3 = self.label_query_proj2(opinion_probs)  # [B, 7, D]
        # attn3 = torch.matmul(q3, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]
        # if mask is not None:
        #     attn3 = attn3.masked_fill(mask.unsqueeze(1), -1e9)
        # w3 = F.softmax(attn3, dim=-1)
        # out3 = torch.matmul(w3, v).mean(dim=1)  # [B, D]
        #
        # # === Fuse attention outputs ===
        # label_out = (out2 + out3) / 2.0  # [B, D]
        # =========================end=============revise fuse part=======================

        gate_input = torch.cat([rating_feature, pooled], dim=-1)
        # gate_input = torch.cat([rating_feature, output], dim=-1)
        gate = self.fuse_gate(gate_input)  # [B, D]
        final_rep = gate * rating_feature + (1 - gate) * pooled
        # final_rep = gate * rating_feature + (1 - gate) * output

        review_score = self.review_score_predictor(final_rep).squeeze(-1)  # [B]

        return review_score


    def forward(self, opinion_probs, opinion_sum_labels, rating_feature, mask=None, use_ground_truth_labels=False, true_labels=None):
        # x shape: [batch_size, num_attributes, input_dim]
        batch_size = opinion_probs.size(0)



        # Project to query, key, value
        # opinion_probs_d = opinion_probs.detach()
        sentiment_vecs = torch.matmul(opinion_sum_labels, self.sentiment_embedding)  # [B, 7, 448]
        # sentiment_vecs_d = torch.matmul(opinion_probs_d, self.sentiment_embedding)
        # sentiment_vecs = torch.cat([opinion_features, sentiment_vecs], dim=-1)  # [B, 7, 2D]

        if mask is not None:  ##do we need two masks??
            sentiment_vecs = sentiment_vecs.masked_fill(mask.unsqueeze(-1), 0.0)
            # sentiment_vecs_d = sentiment_vecs_d.masked_fill(mask.unsqueeze(-1), 0.0)
        # q = self.diff_query_proj(rating_diff.unsqueeze(-1)).unsqueeze(1)  # [B, 1, D]



        q = self.query(sentiment_vecs)
        k = self.key(sentiment_vecs)
        v = self.value(sentiment_vecs)

        # k_d = self.key(sentiment_vecs_d)
        # v_d = self.value(sentiment_vecs_d)

        # diff = rating_diff.unsqueeze(-1)  # [B, 1]
        # q = self.diff_query(diff).unsqueeze(1)  # [B, 1, D]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), -1e9)

        # Calculate attention scores
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_attributes, num_attributes]
        attn_weights = F.softmax(attn_scores, dim=-1) #[batch, 7, 7]

        # Apply attention to values
        weighted = torch.matmul(attn_weights, v)  # [batch_size, num_attributes, input_dim]

        output = weighted + sentiment_vecs  # residual
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_counts = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        else:
            valid_counts = torch.tensor(self.num_attributes, dtype=torch.float32, device=opinion_probs.device).unsqueeze(0).repeat(batch_size, 1)

        pooled = output.sum(dim=1) / valid_counts  # [B, D]
        # output = weighted.squeeze(1) #[B,D]

        # ==========multi-heads multi queries=============
        # === Head 2: label head 1 ===
        # q2 = self.label_query_proj1(opinion_probs)  # [B, 7, D]
        # attn2 = torch.matmul(q2, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]
        # if mask is not None:
        #     attn2 = attn2.masked_fill(mask.unsqueeze(1), -1e9)
        # w2 = F.softmax(attn2, dim=-1)
        # out2 = torch.matmul(w2, v).mean(dim=1)  # [B, D]
        #
        # # === Head 3: label head 2 ===
        # q3 = self.label_query_proj2(opinion_probs)  # [B, 7, D]
        # attn3 = torch.matmul(q3, k.transpose(-2, -1)) / self.scale  # [B, 7, 7]
        # if mask is not None:
        #     attn3 = attn3.masked_fill(mask.unsqueeze(1), -1e9)
        # w3 = F.softmax(attn3, dim=-1)
        # out3 = torch.matmul(w3, v).mean(dim=1)  # [B, D]
        #
        # # === Fuse attention outputs ===
        # label_out = (out2 + out3) / 2.0  # [B, D]
    #=========================end=============revise fuse part=======================

        gate_input = torch.cat([rating_feature, pooled], dim=-1)
        # gate_input = torch.cat([rating_feature, output], dim=-1)
        gate = self.fuse_gate(gate_input)  # [B, D]
        final_rep = gate * rating_feature + (1 - gate) * pooled
        # final_rep = gate * rating_feature + (1 - gate) * output

        review_score = self.review_score_predictor(final_rep).squeeze(-1)  # [B]

        if use_ground_truth_labels and true_labels is not None:
            gt_review_score = self.gt_review_score(true_labels, rating_feature, mask)
            return review_score, gt_review_score

        else:
            return review_score, output


class MMOE_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(MMOE_Classifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)  # 保持维度
        self.linear2 = nn.Linear(input_dim, num_classes)
        self.init_weights()  # 初始化（同原MMOE_MLP）
    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):
        mlp_vector = F.relu(self.linear1(hidden))  # 移除Sigmoid
        logits = self.linear2(mlp_vector)          # 输出4维
        return F.softmax(logits, dim=-1)           # 多分类概率

class MultiTaskModel(nn.Module):
    def __init__(self, num_users, num_items, input_dim, expert_dim, num_experts, sentiment_loss, num_attributes):
        super(MultiTaskModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, dropout=0.5)
        self.num_attributes = num_attributes
        # self.max_label_size = max_label_size
        self.sentiment_loss = sentiment_loss

        # User & Item Embeddings
        self.user_embedding = nn.Embedding(num_users, input_dim)
        self.item_embedding = nn.Embedding(num_items, input_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
#========================weight for total loss ===========================
        self.log_sigma_rating = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_residual = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_opinion = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_sentiment = nn.Parameter(torch.tensor(0.0))

        # Experts with transformer layers
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_dim) for _ in range(num_experts)
        ])

        # Gate for rating prediction
        self.rating_gate = GateNetwork(num_experts, input_dim)

        # Gates for review generation (one per attribute)
        self.review_gates = nn.ModuleList([
            GateNetwork(num_experts, input_dim) for _ in range(num_attributes)
        ])

        # User-Item Interaction Layer
        self.interaction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(input_dim)
            nn.LayerNorm(input_dim)
        )

        # Attribute towers for review generation
        self.attribute_towers = nn.ModuleList([
            MMOE_Classifier(input_dim)
            for _ in range(num_attributes)
        ])
        # self.attribute_towers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(expert_dim, 64),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(64),
        #         nn.Linear(64, 4), #one-hot label: 4
        #         nn.Softmax(dim=-1)
        #     )
        #     for _ in range(num_attributes)
        # ])
        # self.rating_tower = MLP_as_MMOE(input_dim)
        self.rating_tower = MMOE_MLP(input_dim) #输入的shape应该和输入数据最后一个shape一致

        # Attention Network for review score
        # self.embedding_transform = nn.Linear(input_dim, expert_dim)
        # Review attention network
        self.review_attention = ReviewAttentionNetwork(input_dim, num_attributes)
        # self.opinion_aggregator = OpinionSentimentAggregator(input_dim=448)

        # self.review_attention = ReviewTransformerEncoder(input_dim, num_attributes)

        # Review score prediction
        self.review_score = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

######calculate group-item one by one ###########
    def group_mmoe_pred(self, group_ids, item_ids):
        group_logits_list_all = []
        predicted_opinion_sum_labels = []
        predicted_opinion_avg_labels = []

        for b, group in enumerate(group_ids):
            group = torch.tensor(group, device=item_ids.device, dtype=torch.long)
            g_emb = self.user_embedding(group)  # [G_i, D]
            num_user = g_emb.shape[0]
            # g_emb_mean = g_emb.mean(dim=0, keepdim=True)  # [1, D]
            # i_emb = self.item_embedding(item_ids[b])  # [1, D] since we're processing one item
            # gi_rep = torch.cat([g_emb_mean, i_emb.unsqueeze(0)], dim=0)  # [2, 1, D]
            i_emb = self.item_embedding(item_ids[b].repeat(len(group)))  # [G_i, D]
            gi_rep = torch.cat([g_emb.unsqueeze(0), i_emb.unsqueeze(0)], dim=0)  # [2, G_i, D]
            # gi_rep = self.interaction_layer(gi_rep)  # [2, G_i, D]
            gi_rep = self.pos_encoder(gi_rep)  # [2, G_i, D]
            gi_rep = gi_rep.mean(dim=0)  # [G_i, D]

            expert_outputs = [expert(gi_rep) for expert in self.experts]  # list of [G_i, D]
            expert_outputs = torch.stack(expert_outputs, dim=1)  # [G_i, num_experts, D]

            attr_logits = []
            for i, gate in enumerate(self.review_gates):
                gate_weights = gate(gi_rep)  # [G_i, num_experts]
                attr_expert_out = torch.einsum("be,bed->bd", gate_weights, expert_outputs)  # [G_i, D]
                logits = self.attribute_towers[i](attr_expert_out)  # [G_i, 4]
                attr_logits.append(logits)

            attr_logits = torch.stack(attr_logits, dim=1)  # [G_i, 7, 4]
            group_logits_list_all.append(attr_logits)
            group_probs = F.softmax(attr_logits, dim=-1)  # [G_i, 7, 4]

            # avg = group_probs.mean(dim=0) #[7,4]  #avg_probs to be predicted one-hot label probs
            # num_user = group_probs.shape[0]
            multi_avg = num_user * group_probs
            # predicted_opinion_sum_labels.append(summed)
            predicted_opinion_sum_labels.append(multi_avg)
            predicted_opinion_avg_labels.append(group_probs)

        predicted_opinion_sum_labels = torch.stack(predicted_opinion_sum_labels, dim=0)  # [B, 7, 4]
        predicted_opinion_avg_labels = torch.stack(predicted_opinion_avg_labels, dim=0) #[B,7,4]
        return predicted_opinion_sum_labels, predicted_opinion_avg_labels

    def forward(self, user_ids, item_ids, group_ids, ratings, opinion_labels, opinion_sum_labels, processed_labels, is_predicted_opinions=True):
        # === Convert User-Item Indices to Embeddings ===
        group_lengths = []
        for b, group in enumerate(group_ids):
            group_lengths.append(len(group))
        user_emb = self.user_embedding(user_ids.unsqueeze(0))  # Shape: `[1, batch_size, emb_dim]`
        item_emb = self.item_embedding(item_ids.unsqueeze(0))  # Shape: `[1, batch_size, emb_dim]`
        # predicted_opinion_sum_labels, pred_group_labels = self.group_mmoe_pred(group_ids, item_ids)

        group_emb = [self.user_embedding(torch.tensor(g, device=item_ids.device)).mean(dim=0) for g in
                     group_ids]  # list of [D]
        group_emb = torch.stack(group_emb, dim=0).unsqueeze(0) #[1, batch_size,448]

        # group_emb = self.user_embedding(group_ids)  # [B, G, D]
        # group_emb = group_emb.mean(dim=1)

        # user_item_representation = user_emb * item_emb #[batch_size, 7*64]
        user_item_representation = torch.cat([user_emb, item_emb], dim=0)  # [2, batch, 448]
        group_item_representation = torch.cat([group_emb, user_emb], dim=0)
        #user-item interaction
        user_item_rep = self.interaction_layer(user_item_representation) #[2,512,448]
        group_item_rep = self.interaction_layer(group_item_representation)
        # user_item_representation = user_item_representation.reshape(512, 448, -1)  # [2,512,448]

        user_item_rep = self.pos_encoder(user_item_rep)  # [2, batch_sie, 448]
        group_item_rep = self.pos_encoder(group_item_rep)
        mean_pool = user_item_rep.mean(dim=0)  # [batch, emsize]
        group_mean_pool = group_item_rep.mean(dim=0)

        #==============user-item for rating prediction============
        expert_outputs = [expert(mean_pool) for expert in self.experts]  # list 3: Each: [batch_size, expert_dim]

        # # Stack and reshape back
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, expert_dim]
# Rating Prediction Task
        rating_gate_weights = self.rating_gate(mean_pool)  # torch.Size([512, 5])
        rating_expert_output = torch.einsum("be,bed->bd", rating_gate_weights, expert_outputs) #[batch_size, exp_dim]
        predicted_rating_score = self.rating_tower(rating_expert_output) #[batch_size,1]
        # print(predicted_rating_score)


        if is_predicted_opinions:
            # 将one-hot opinion labels转换为类别索引 [batch_size, num_attributes]
            # true_label_indices = torch.argmax(batch_weighted_opinions, dim=-1) #[batch, 7]
            # opinion的预测分为两个分支：user-item(用于eval和test做比较); group-item(用于训练)

            user_logits_list = []
            user_embeddings_list = []
            #===========user-item prediction=============
            for i, gate in enumerate(self.review_gates):
                gate_weights = gate(mean_pool)  # user-item interaction

                attribute_expert_output = torch.einsum("be,bed->bd", gate_weights, expert_outputs) #[batch, 448]



                logits = self.attribute_towers[i](attribute_expert_output) #[512, 4]


                user_logits_list.append(logits)
                user_embeddings_list.append(attribute_expert_output)
            user_stacked_probs = torch.stack(user_logits_list, dim=1)
            user_attribute_probs = F.softmax(user_stacked_probs, dim=-1)  # predicted user-item one-hot labels

            #==============group-item prediction==============
            group_logits_list = []
            group_embeddings_list = []
            gp_expert_outputs = [expert(group_mean_pool) for expert in self.experts]  # list 3: Each: [batch_size, expert_dim]

            # # Stack and reshape back
            gp_expert_outputs = torch.stack(gp_expert_outputs, dim=1)  # [batch_size, num_experts, expert_dim]

            for i, gate in enumerate(self.review_gates):
            # gate_weights = gate(group_item)
                gate_weights = gate(group_mean_pool)  # user-item interaction

                attribute_expert_output = torch.einsum("be,bed->bd", gate_weights, gp_expert_outputs) #[batch, 448]



                logits = self.attribute_towers[i](attribute_expert_output) #[512, 4]

            # logits_list.append(F.pad(logits, (0, max_label_size - logits.size(1)), mode="constant", value=0))
                group_logits_list.append(logits)
                group_embeddings_list.append(attribute_expert_output)


            group_stacked_probs = torch.stack(group_logits_list, dim=1)
            # group_attribute_probs = F.softmax(group_stacked_probs, dim=-1)  #predicted one-hot labels
            pred_group_labels = F.softmax(group_stacked_probs, dim=-1)
            group_lengths = torch.tensor(group_lengths, device=item_ids.device, dtype=torch.float32)  # [B]
            predicted_opinion_sum_labels = pred_group_labels * group_lengths.view(-1, 1, 1)  # [B, 7, 4]

            opinion_mask = (processed_labels == -1)

            # rating_diff = ratings - predicted_rating_score.detach()

            pred_user_score = self.review_attention.user_forward(
                opinion_probs=user_attribute_probs,
                # opinion_sum_labels=opinion_sum_labels,
                # opinion_features=stacked_embeddings,
                rating_feature=rating_expert_output,
                # rating_diff = rating_diff,
                # true_labels=opinion_labels,  # gt_group_labels
                mask=opinion_mask,  ##need to be revised: this mask is from gt_group_labels, not user_opinion_label(6.20)
                # use_ground_truth_labels=False  # 是否用true labels做sentiment vec
            )

            user_final_score = predicted_rating_score + 0.8 * pred_user_score

            predicted_review_score, gt_review_score = self.review_attention(
                # opinion_probs=group_attribute_probs,
                opinion_probs=pred_group_labels,
                opinion_sum_labels=predicted_opinion_sum_labels,
                # opinion_features=stacked_embeddings,
                rating_feature=rating_expert_output,
                # rating_diff=rating_diff, ##useless 6.19
                true_labels=opinion_sum_labels, #gt_group_labels
                mask=opinion_mask,
                use_ground_truth_labels=True  #是否用true labels做sentiment vec
            )

            # final_score = predicted_rating_score + 0.8 * gt_review_score
            final_score = predicted_rating_score + 0.8 * predicted_review_score #权重高predicted_rating_score变差
        else:
            attribute_probs = None
            predicted_review_score = torch.zeros_like(predicted_rating_score)
            final_score = predicted_rating_score


        return predicted_rating_score, pred_group_labels, user_attribute_probs, user_final_score, predicted_review_score, final_score, gt_review_score, predicted_opinion_sum_labels

    def calculate_loss(self, user_ids, item_ids, group_ids, ratings, gt_user_labels, group_sum_labels, gt_group_labels, group_processed_labels, is_predicted_opinions=True):
        ##we need compare performance when using user-item embs and group-item embs
        #predicted_ratings: predicted from user-item embedding;
        # predicted_opinions: predicted labels from group embeddings
        #predicted_review_score: score from predicted lables through attention network
        #final_score: final ratings adjusted by predicted opinions
        #gt_review_score: score from true labels trhough attention networks
        predicted_ratings, predicted_opinions, _, _, predicted_review_score, final_score, gt_review_score, pred_opinion_sum_labels = self.forward(user_ids, item_ids, group_ids, ratings, gt_group_labels, group_sum_labels, group_processed_labels, is_predicted_opinions)
        rating_loss = F.mse_loss(predicted_ratings, ratings)
        # print("Any NaN in target:", torch.isnan(final_score).any().item())

        if is_predicted_opinions:
            rating_review_loss = F.mse_loss(final_score, ratings)
            device = user_ids.device
            user_opinion_loss = 0.0
            group_opinion_loss = 0.0
            valid_opinion_count = 0
            lambda_user = 0.4

            for i in range(self.num_attributes):
                # 获取当前attribute的所有样本标签 [batch_size, 4]
                curr_user_labels = gt_user_labels[:, i, :]
                curr_group_labels = gt_group_labels[:, i, :]

                # 检查哪些样本有有效标签 (非全零)
                user_valid_mask = (curr_user_labels.sum(dim=1)) > 0  # [batch_size]
                group_valid_mask = (curr_group_labels.sum(dim=1)) > 0

                if user_valid_mask.any():
                # 只计算有效样本的user opinion loss和group loss
                    user_valid_pred = predicted_opinions[:, i, :][user_valid_mask]
                    group_valid_pred = predicted_opinions[:, i, :][group_valid_mask]
                    user_valid_true = torch.argmax(curr_user_labels[user_valid_mask], dim=1)
                    group_valid_true = torch.argmax(curr_group_labels[group_valid_mask], dim=1)

                    user_opinion_loss += F.cross_entropy(user_valid_pred, user_valid_true)
                    group_opinion_loss += F.cross_entropy(group_valid_pred, group_valid_true)
                    valid_opinion_count += 1

                # 平均有效的attribute loss

            if valid_opinion_count > 0:
                user_opinion_loss /= valid_opinion_count
                group_opinion_loss /= valid_opinion_count
                opinion_loss = (1 - lambda_user) * user_opinion_loss + lambda_user * group_opinion_loss

            else:
                opinion_loss = 0.0


            #===================sentiment loss============
            if self.sentiment_loss > 0:

                sentiment_vec_pred = torch.matmul(predicted_opinions,
                                                  self.review_attention.sentiment_embedding)  # [B, 7, D]
                sentiment_vec_gt = torch.matmul(gt_group_labels.float(),
                                                self.review_attention.sentiment_embedding)  # [B, 7, D]
                sentiment_vec_sum = torch.matmul(group_sum_labels.float(),
                                                self.review_attention.sentiment_embedding)  # [B, 7, D]

                # 扁平化所有有效位置进行比较
                valid_mask = (gt_group_labels.sum(dim=-1) > 0)  # [B, 7]
                # rating_diff = ratings - predicted_ratings.detach()


                cosine_loss = F.cosine_embedding_loss(
                    sentiment_vec_pred[valid_mask],
                    sentiment_vec_gt[valid_mask],
                    torch.ones(valid_mask.sum(), device=sentiment_vec_gt.device)
                )
                cosine_sum_loss = F.cosine_embedding_loss(sentiment_vec_pred[valid_mask],
                    sentiment_vec_sum[valid_mask],
                    torch.ones(valid_mask.sum(), device=sentiment_vec_gt.device))
                #==================head2 and 3 loss============================
                opinion_sentiment_vecs = sentiment_vec_gt.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

                target_sentiment = opinion_sentiment_vecs.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B, D]
                #=============label_sum for supervision====================

                # cosine_sentiment_loss = F.cosine_embedding_loss(
                #     label_attn, target_sentiment,
                #     torch.ones(label_attn.size(0), device=label_attn.device)
                # )
                #==============loss for predicted_sum_opinion_labels========
                count_mask = (group_sum_labels.sum(dim=-1) > 0)  # [B, 7]
                masked_pred = pred_opinion_sum_labels[count_mask]  # [N, 4]
                masked_gt = group_sum_labels[count_mask].float()  # [N, 4]
                opinion_count_loss = F.mse_loss(masked_pred, masked_gt)

                delta_target = ratings - predicted_ratings.detach()  # 真实评分 - 结构预测评分
                # residual_loss = F.mse_loss(gt_review_score, delta_target)
                residual_loss = F.mse_loss(predicted_review_score, delta_target)
                sentiment_align_loss = F.mse_loss(predicted_review_score, gt_review_score)

                direction_labels = torch.sign(delta_target.detach())  # [B], values in {-1, 0, 1}
                direction_labels[direction_labels == 0] = 1  # prevent 0 vector (treat as neutral-positive)
                cosine_dir_loss = F.cosine_embedding_loss(
                    predicted_review_score.unsqueeze(-1),  # [B, 1]
                    delta_target.unsqueeze(-1),  # [B, 1]
                    target=direction_labels.float(),  # [B]
                )  # direction loss

                # total_loss = rating_review_loss + 0.8 * residual_loss + 1.0 * opinion_loss + 10 * cosine_dir_loss
                # total_loss = rating_review_loss + 0.8 * residual_loss + 5.0 * opinion_loss + 0.5 * cosine_dir_loss + 0.5 * cosine_sentiment_loss
                # total_loss = rating_review_loss + 0.8 * residual_loss + 1.5 * opinion_loss + opinion_count_loss + 15 * cosine_sum_loss + 0.4 * sentiment_align_loss
                # total_loss = rating_review_loss + 0.4 * residual_loss + 10 * opinion_loss
                # total_loss = rating_review_loss + 1.5 * opinion_loss + 1.5 * opinion_count_loss + 15 * cosine_sum_loss + 0.4 * sentiment_align_loss
                total_loss = rating_review_loss + 1.5 * opinion_loss +  opinion_count_loss + 15 * cosine_sum_loss +1.5 * sentiment_align_loss



            else:
                total_loss = rating_review_loss + opinion_loss
        # else:
        #     total_loss = rating_loss

            return total_loss
        #     opinion_loss = 0.0
        #     for i in range(self.num_attributes):
        #         opinion_loss += F.cross_entropy(predicted_opinions[:, i, :], opinion_labels[:, i])
        #     total_loss = rating_loss + opinion_loss
        # else:
        #     total_loss = rating_loss
        # # else: opinion_loss = 0
        #
        # return total_loss
    def rating_forward(self, user_ids, item_ids):
        # === Convert User-Item Indices to Embeddings ==
        user_emb = self.user_embedding((user_ids.unsqueeze(0)))  # Shape: `[1, batch_size, emb_dim]`
        item_emb = self.item_embedding((item_ids.unsqueeze(0)))
        user_item_representation = torch.cat([user_emb, item_emb], dim=0)  # [2, batch, emsize]
        user_item_rep = self.interaction_layer(user_item_representation)  # [2,512,448]

        # ==========================================================================
        # Apply Transformer Encoder

        user_item_rep = self.pos_encoder(user_item_rep)  # [2, batch_sie, 448]

        mean_pool = user_item_rep.mean(dim=0)  # [batch, emsize]

        expert_outputs = [expert(mean_pool) for expert in self.experts]  # list 5: Each: [batch_size, expert_dim]
        # # Stack and reshape back
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, expert_dim]

        # Rating Prediction Task
        rating_gate_weights = self.rating_gate(mean_pool)  # torch.Size([512, 5])
        rating_expert_output = torch.einsum("be,bed->bd", rating_gate_weights, expert_outputs)  # [batch_size, exp_dim]
        predicted_rating_score = self.rating_tower(rating_expert_output)  # [batch_size,1]
        return predicted_rating_score

    def predict_opinions(self, user_ids, item_ids, group_ids, ratings, gt_user_labels, group_sum_labels, gt_group_labels, group_processed_labels):
        predicted_ratings, predicted_labels, user_pred_labels, _, predicted_review_score, final_score, gt_review_score, _ = self.forward(user_ids, item_ids, group_ids, ratings, gt_group_labels, group_sum_labels, group_processed_labels,
                                                                             is_predicted_opinions=True)
        rating_review_loss = F.mse_loss(final_score, ratings)
        device = user_ids.device
        user_opinion_loss = 0.0
        group_opinion_loss = 0.0
        valid_opinion_count = 0
        lambda_user = 0.4

        for i in range(self.num_attributes):
            # 获取当前attribute的所有样本标签 [batch_size, 4]
            curr_user_labels = gt_user_labels[:, i, :]
            curr_group_labels = gt_group_labels[:, i, :]

            # 检查哪些样本有有效标签 (非全零)
            user_valid_mask = (curr_user_labels.sum(dim=1)) > 0  # [batch_size]
            group_valid_mask = (curr_group_labels.sum(dim=1)) > 0

            if user_valid_mask.any():
                # 只计算有效样本的user opinion loss和group loss
                user_valid_pred = predicted_labels[:, i, :][user_valid_mask]
                group_valid_pred = predicted_labels[:, i, :][group_valid_mask]
                user_valid_true = torch.argmax(curr_user_labels[user_valid_mask], dim=1)
                group_valid_true = torch.argmax(curr_group_labels[group_valid_mask], dim=1)

                user_opinion_loss += F.cross_entropy(user_valid_pred, user_valid_true)
                group_opinion_loss += F.cross_entropy(group_valid_pred, group_valid_true)
                valid_opinion_count += 1

            # 平均有效的attribute loss

        if valid_opinion_count > 0:
            user_opinion_loss /= valid_opinion_count
            group_opinion_loss /= valid_opinion_count
            opinion_loss = (1 - lambda_user) * user_opinion_loss + lambda_user * group_opinion_loss

        else:
            opinion_loss = 0.0

        # for i in range(self.num_attributes):
        #     # 获取当前attribute的所有样本标签 [batch_size, 4]
        #     curr_labels = opinion_labels[:, i, :]
        #     # 检查哪些样本有有效标签 (非全零)
        #     valid_mask = (curr_labels.sum(dim=1)) > 0  # [batch_size]
        #
        #     if valid_mask.any():
        #         # 只计算有效样本的loss
        #         valid_pred = predicted_labels[:, i, :][valid_mask]
        #         valid_true = torch.argmax(curr_labels[valid_mask], dim=1)
        #
        #         opinion_loss += F.cross_entropy(valid_pred, valid_true)
        #         valid_opinion_count += 1

            # 平均有效的attribute loss

        # if valid_opinion_count > 0:
        #     opinion_loss /= valid_opinion_count
        #
        # else:
        #     opinion_loss = 0.0
        if self.sentiment_loss > 0:
            if self.sentiment_loss > 0:
                sentiment_vec_pred = torch.matmul(predicted_labels,
                                                  self.review_attention.sentiment_embedding)  # [B, 7, D]
                sentiment_vec_gt = torch.matmul(gt_group_labels.float(),
                                                self.review_attention.sentiment_embedding)  # [B, 7, D]

                # 扁平化所有有效位置进行比较
                valid_mask = (gt_group_labels.sum(dim=-1) > 0)  # [B, 7]
                cosine_loss = F.cosine_embedding_loss(
                    sentiment_vec_pred[valid_mask],
                    sentiment_vec_gt[valid_mask],
                    torch.ones(valid_mask.sum(), device=sentiment_vec_gt.device)
                )
            #
            # valid_mask = (gt_group_labels.sum(dim=-1) > 0)  # [B, 7]
            # gt_sentiment_vecs = gt_sentiment_vecs.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            # avg_sentiment_vec = gt_sentiment_vecs.sum(dim=1) / (
            #     valid_mask.sum(dim=1).clamp(min=1).unsqueeze(-1))  # [B, D]

            delta_target = ratings - predicted_ratings.detach()  # 真实评分 - 结构预测评分
            # residual_loss = F.mse_loss(gt_review_score, delta_target)
            residual_loss = F.mse_loss(predicted_review_score, delta_target)
            sentiment_align_loss = F.mse_loss(predicted_review_score, gt_review_score)

            # target_review_score = self.review_attention.review_score_predictor(avg_sentiment_vec).squeeze(-1)
            # sentiment_align_loss = F.mse_loss(predicted_review_score, target_review_score)



            # sigma_rating = F.softplus(self.log_sigma_rating)
            # sigma_residual = F.softplus(self.log_sigma_residual)
            # sigma_opinion = F.softplus(self.log_sigma_opinion)
            # sigma_sentiment = F.softplus(self.log_sigma_sentiment)

            # sentiment_align_loss = F.mse_loss(predicted_review_score, target_review_score)
            total_loss = rating_review_loss + 1.0 * opinion_loss + sentiment_align_loss
            # total_loss = rating_review_loss + 0.4 * residual_loss + 7 * opinion_loss + 10 * sentiment_align_loss

            # total_loss = rating_review_loss + 1.0 * residual_loss + 0.8 * opinion_loss + 0.0 * sentiment_align_loss


        else:
            total_loss = rating_review_loss + opinion_loss
        return rating_review_loss, total_loss, predicted_labels, user_pred_labels #[batch_size, max_item_num]

    def predict(self, user_ids, item_ids):


        predicted_ratings = self.rating_forward(user_ids, item_ids)
        # predicted_ratings, _, _, _ = self.forward(user_ids, item_ids, opinion_weights)
        return predicted_ratings #[batch_size, max_item_num]

    def predict_rating_with_reviews(self, user_ids, item_ids, group_ids, ratings, opinion_labels, group_sum_labels, processed_labels, is_predicted_opinions=True):
        predicted_ratings, group_pred_labels, user_pred_labels, user_final_score, _, final_score, _, _ = self.forward(user_ids, item_ids, group_ids, ratings, opinion_labels, group_sum_labels, processed_labels, is_predicted_opinions=True)
        return predicted_ratings, group_pred_labels, user_pred_labels, user_final_score, final_score
