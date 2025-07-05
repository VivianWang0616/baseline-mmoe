import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
from recbole.config import Config
from recbole.utils import EvaluatorType, calculate_valid_score

from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from metircs import root_mean_square_error, mean_absolute_error
from peter_dataloader import now_time
import seaborn as sns
import numpy as np

class group_MMOE_trainer:

    """
    Trainer class for running experiments with ExpGCN and other baselines.
    Supports rating prediction (MSELoss) and attribute-opinion ranking (CrossEntropyLoss).
    Uses RecBole Evaluator for Recall@K, NDCG@K, and MSE.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, device, save_dir="./saved_models"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir

        # Initialize best model tracking
        self.best_valid_score = float("inf")  # Track highest Recall@10
        self.best_valid_result = None
        self.best_model_path = os.path.join(self.save_dir, f"best_{self.model.__class__.__name__}.pth")

        os.makedirs(self.save_dir, exist_ok=True)  # Ensure save directory exists

    def preprocess_labels(self, one_hot_labels):
        """
        将one-hot标签转换为索引，并正确处理全零标签
        参数:
            one_hot_labels: [batch_size, num_attributes, 4] (one-hot格式)
        返回:
            label_indices: [batch_size, num_attributes] (有效标签为0-3，全零标签为-1)
        """
        # 步骤1：创建全零标签的掩码
        is_valid = (one_hot_labels.sum(dim=-1)) > 0  # [batch_size, num_attributes]

                    # 步骤2：初始化结果张量（全填充-1）
        label_indices = torch.full_like(is_valid, -1, dtype=torch.long)  # [batch_size, num_attributes]

        # 步骤3：仅对有效标签计算argmax
        if is_valid.any():
            valid_labels = one_hot_labels[is_valid]  # [num_valid, 4]
        label_indices[is_valid] = torch.argmax(valid_labels, dim=1)

        return label_indices

    def aggregate_group_labels(self, group_labels):
        """
        Args:
            group_labels: list of B tensors, each [G, 7, 4]
        Returns:
            aggregated: Tensor [B, 7, 4] one-hot (or all-zero if no label)
        """
        aggregated = []

        for g_labels in group_labels:  # [G, 7, 4]
            # label_sum = g_labels.sum(dim=0)  # [7, 4]
            label_sum = torch.stack(g_labels).sum(dim=0)  # [7, 4]
            attr_valid_mask = (label_sum.sum(dim=-1) > 0)  # [7] 是否有label

            # 生成全零标签
            one_hot = torch.zeros_like(label_sum)  # [7, 4]

            # 只对有效属性进行投票
            if attr_valid_mask.any():
                max_indices = label_sum.argmax(dim=-1)  # [7]
                for i in range(7):
                    if attr_valid_mask[i]:
                        one_hot[i, max_indices[i]] = 1.0  # 只赋值有效位置

            aggregated.append(one_hot.unsqueeze(0))  # [1, 7, 4]

        return torch.cat(aggregated, dim=0)  # [B, 7, 4]

    # def aggregate_group_labels(self, group_labels):
    #     """
    #     Args:
    #         group_labels: list of B tensors, each [G, 7, 4]
    #     Returns:
    #         aggregated_labels: Tensor [B, 7, 4] one-hot (majority vote)
    #     """
    #     aggregated = []
    #     for g_labels in group_labels:  # [G, 7, 4]
    #         G = g_labels.size(0)
    #         # sum over G users: [7, 4]
    #         label_sum = g_labels.sum(dim=0)
    #
    #         # vote: 取最大值位置，one-hot
    #         max_indices = label_sum.argmax(dim=-1)  # [7]
    #         one_hot = torch.zeros_like(label_sum)
    #         one_hot.scatter_(1, max_indices.unsqueeze(1), 1.0)  # [7, 4]
    #         aggregated.append(one_hot.unsqueeze(0))  # [1, 7, 4]
    #
    #     return torch.cat(aggregated, dim=0)  # [B, 7, 4]

    def train(self, num_epochs, is_predicted_opinion=False):
        """
        Train the model for a given number of epochs and save the best model based on Recall@10.
        """
        self.model.train()
        plot_train_loss = []
        plot_val_loss = []
        trigger_times = 0

        for epoch in range(num_epochs):
            total_loss = []

            count = 0
            # self.peter_train()

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training"):
                user_ids = batch['user_idx'].to(self.device)  # [B]
                item_ids = batch['item_idx'].to(self.device)  # [B]
                ratings = batch['rating'].to(self.device)  # [B]
                opinion_labels = batch['user_opinion_label'].to(self.device)  # [B, 7, 4]
                # single_label_indices = self.preprocess_labels(opinion_labels)  # [B, 7] (optional)

                group_ids = batch['group_user_ids']  # list of B lists
                # group_ids = torch.stack([torch.tensor(g, device=item_ids.device) for g in group_ids], dim=0)  # [B, G]
                # gt_group_labels = torch.stack(batch['group_labels']).to(self.device)  # [B, 7, 4]
                group_labels = batch['group_labels']
                gt_group_labels = self.aggregate_group_labels(group_labels).to(self.device)  # [B, 7, 4]
                group_sum_labels = batch['group_sum_label'].to(self.device)  # [B, 7, 4]

                group_label_indices = self.preprocess_labels(gt_group_labels)

            # for ratings_data, opinion_data in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training"):
            #     ratings_data = ratings_data.to(self.device) #(1024,3)
            #     opinion_data = opinion_data.to(self.device) #(1024,10,176)
            #     batch_label_indices = self.preprocess_labels(opinion_data) #[batch, num_attributes] 返回非零one-hot label argmax indice
            #
            #     #
            #     user_ids = ratings_data[:, 0].long()
            #     item_ids = ratings_data[:, 1].long()  # batch_size
            #     ratings = ratings_data[:, 2]

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # self.optimizer.zero_grad()
                self.optimizer.step()

                ####train和val计算loss均用group_labels

                loss = self.model.calculate_loss(user_ids, item_ids, group_ids, ratings, opinion_labels, group_sum_labels, gt_group_labels, group_label_indices, is_predicted_opinions=True)

                loss.backward()
                #===================
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss.append(loss.item())   #plot train loss
                # total_loss.append(0)
                count+=1

            avg_loss = sum(total_loss) / len(self.train_loader)
            # self.scheduler.step(avg_loss)  #reduce lr
            self.scheduler.step()
            print('Learning rate set to {:2.8f}'.format(self.scheduler.get_last_lr()[0]))

            if is_predicted_opinion:
                org_rating_loss, val_opinion_loss, valid_results = self.evaluate(self.val_loader, is_predicted_opinion=True)
            else:
                org_rating_loss, valid_results = self.evaluate(self.val_loader, is_predicted_opinion=False)
            rating_review_loss = calculate_valid_score(valid_results, "Val Loss")
            valid_score = calculate_valid_score(valid_results, "Accuracy for group opinions labels")
            acc_user = calculate_valid_score(valid_results, "Accuracy for user opinions labels")
            # rating_review_loss = calculate_valid_score(valid_results, "original rating loss")
            # plot_train_loss.append(avg_loss)
            # plot_val_loss.append(val_rating_loss)
            # print(
            #     now_time() + f"Epoch {epoch + 1} Training " + 'valid loss {:4.4f} on validation'.format(val_rating_loss)
            # )


            print(f"Epoch {epoch+1}: Train Loss(rating+attr) = {avg_loss:.4f}, Val Rating Loss = {org_rating_loss:.4f}")
            print("🎯 Final Eval Results:", valid_results)
            # === Save Best Model Based on MSE loss ===
            if rating_review_loss < self.best_valid_score and rating_review_loss < org_rating_loss:
                self.best_valid_score = rating_review_loss
                # self.best_valid_score = valid_score
                self.best_valid_result = valid_results
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"✅ Best model saved at {self.best_model_path} with best validation score of {self.best_valid_score}")

            else:
                trigger_times += 1
                if trigger_times >= 5:
                    print("Early stopping!")
                    print(trigger_times)
                    break
        save_dir = "plots/mmoe_loss.png"
        self.plot_loss_curves(plot_train_loss, plot_val_loss, save_path=save_dir)
    def evaluate(self, data_loader, is_predicted_opinion=False):
        """
        Evaluate the model using RecBole's Evaluator.
        Returns the validation loss and evaluation metrics.
        """
        self.model.eval()
        total_rating_loss = []
        ratings_opinions = []
        total_losses = []
        total_attr_loss = 0
        total_opinion_loss = []
        # recall_at_10, recall_at_20 = 0, 0
        # ndcg_at_10, ndcg_at_20 = 0, 0
        total_mae = 0
        # count = 0
        total_group_attr_acc = 0
        total_user_attr_acc = 0


        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                user_ids = batch['user_idx'].to(self.device)  # [B]
                item_ids = batch['item_idx'].to(self.device)  # [B]
                ratings = batch['rating'].to(self.device)  # [B]
                opinion_labels = batch['user_opinion_label'].to(self.device)  # [B, 7, 4]
                # batch_label_indices = self.preprocess_labels(opinion_labels)  # [B, 7] (optional)

                group_ids = batch['group_user_ids']  # list of B lists
                # group_ids = torch.stack([torch.tensor(g, device=item_ids.device) for g in group_ids], dim=0)  # [B, G]
                # gt_group_labels = torch.stack(batch['group_labels']).to(self.device) # [B, 7, 4]
                group_sum_labels = batch['group_sum_label'].to(self.device)
                group_labels = batch['group_labels']
                gt_group_labels = self.aggregate_group_labels(group_labels).to(self.device)  # [B, 7, 4]
                batch_label_indices = self.preprocess_labels(gt_group_labels)

                predicted_ratings = self.model.predict(user_ids, item_ids) #[batch_size, max_num_items]


                rating_loss = F.mse_loss(predicted_ratings, ratings)
                # MAE = F.l1_loss(predicted_ratings, ratings, reduction='mean')
                # total_rating_loss += rating_loss.item()
                total_rating_loss.append(rating_loss.item())
                # total_mae += MAE.item()




 #=============================attributes evaluation =================================
                if is_predicted_opinion:
                    rating_review_loss, total_loss, group_predicted_labels, user_predicted_labels = self.model.predict_opinions(user_ids, item_ids, group_ids, ratings, opinion_labels, group_sum_labels, gt_group_labels, batch_label_indices)

                # true_opinions = opinion_data[:, :, 2:, :]

                    ratings_opinions.append(rating_review_loss.item())
                    total_losses.append(total_loss.item())

                    #we calculate accuracy for user-item opinion labels, not group labels
                    group_attr_accuracy = self.compute_accuracy(gt_group_labels, group_predicted_labels)
                    # group_attr_accuracy = self.compute_accuracy(opinion_labels, group_predicted_labels)
                    user_attr_accuracy = self.compute_accuracy(opinion_labels, user_predicted_labels)
                    total_group_attr_acc += group_attr_accuracy
                    total_user_attr_acc += user_attr_accuracy

                else:
                    total_group_attr_acc = 0
                    total_user_attr_acc = 0
                    ratings_opinions = 0


            # Compute accuracy for attribute prediction by checking position match



            avg_opinion_loss = sum(ratings_opinions) / len(data_loader)  #each epoch
            # avg_total_loss = sum(total_losses) / len(data_loader)
            avg_rating_loss = sum(total_rating_loss) / len(data_loader)
            avg_user_label_acc = total_user_attr_acc / len(data_loader)
            avg_group_label_acc = total_group_attr_acc / len(data_loader)
            # avg_mae = total_mae / len(data_loader)

        # eval_results = collector.get_result()
        eval_results = {
            # "Recall@10": recall_at_10 / len(data_loader),
            # "Recall@20": recall_at_20 / len(data_loader),
            # "NDCG@10": ndcg_at_10 / len(data_loader),
            # "NDCG@20": ndcg_at_20 / len(data_loader),
            "original rating loss": avg_rating_loss,
            "Val Loss": avg_opinion_loss,
            # "Rating plus opinions": avg_total_loss,
            # "MAE Loss": avg_mae,
            "Accuracy for user opinions labels": avg_user_label_acc,
            "Accuracy for group opinions labels": avg_group_label_acc
        }
        if is_predicted_opinion:
            return avg_rating_loss, avg_opinion_loss, eval_results
        else:
            return avg_rating_loss, eval_results  #plot val loss compared to train loss


    def test(self, is_predicted_opinion=False):
        """
        Load the best model and perform final testing.
        """
        print(f"🔍 Loading best model from {self.best_model_path} for final testing...")
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        # total_rating_loss = 0
        # recall_at_10, recall_at_20 = 0, 0
        # ndcg_at_10, ndcg_at_20 = 0, 0
        # count = 0

        total_group_attr_acc = 0
        total_user_attr_acc = 0

        total_mae = 0
        total_rmse = 0
        total_mae_r = 0
        total_mae_g_r_w_r = 0
        total_mae_u_r_w_r = 0
        total_rmse_r=0
        total_rmse_g_r_w_r = 0
        total_rmse_u_r_w_r = 0

        all_gt = []
        all_pred = []
        group_pred_ratings = []
        true_ratings = []


        with torch.no_grad():
            # RMSE, MAE = self.peter_test()
            for batch in tqdm(self.test_loader, desc="Testing"):
                user_ids = batch['user_idx'].to(self.device)  # [B]
                item_ids = batch['item_idx'].to(self.device)  # [B]
                ratings = batch['rating'].to(self.device)  # [B]
                opinion_labels = batch['user_opinion_label'].to(self.device)  # [B, 7, 4]
                # batch_label_indices = self.preprocess_labels(opinion_labels)  # [B, 7] (optional)

                group_ids = batch['group_user_ids']  # list of B lists

                # group_ids = torch.stack([torch.tensor(g, device=item_ids.device) for g in group_ids], dim=0)  # [B, G]

                # gt_group_labels = torch.stack(batch['group_labels']).to(self.device)
                group_sum_labels = batch['group_sum_label'].to(self.device)
                group_labels = batch['group_labels']
                gt_group_labels = self.aggregate_group_labels(group_labels).to(self.device)  # [B, 7, 4]
                batch_label_indices = self.preprocess_labels(gt_group_labels)



                predicted_ratings = self.model.predict(user_ids, item_ids)  # [batch_size, max_num_items]


                # true_ratings.extend(ratings.tolist())
                pred_ratings = [(r, p) for (r, p) in zip(ratings.tolist(), predicted_ratings.tolist())]
                rmse = root_mean_square_error(pred_ratings, 5, 1)
                total_rmse += rmse
                mae = mean_absolute_error(pred_ratings, 5, 1)
                total_mae += mae

    #===================opinion test==========================
                if is_predicted_opinion:
                    predicted_ratings_org, group_pred_labels, user_pred_labels, user_rating_with_reviews, group_rating_with_reviews = self.model.predict_rating_with_reviews(user_ids, item_ids, group_ids, ratings, gt_group_labels, group_sum_labels, batch_label_indices)

                    pred_ratings_org = [(r, p) for (r, p) in zip(ratings.tolist(), predicted_ratings_org.tolist())]
                    g_r_w_r = [(r, p) for (r, p) in zip(ratings.tolist(), group_rating_with_reviews.tolist())]
                    u_r_w_r = [(r, p) for (r, p) in zip(ratings.tolist(), user_rating_with_reviews.tolist())]

                    group_pred_ratings.extend(group_rating_with_reviews.tolist())
                    true_ratings.extend(ratings.tolist())

                    rmse_r = root_mean_square_error(pred_ratings_org, 5, 1)
                    rmse_g_r_w_r = root_mean_square_error(g_r_w_r, 5, 1)
                    rmse_u_r_w_r = root_mean_square_error(u_r_w_r, 5, 1)
                    total_rmse_r += rmse_r
                    total_rmse_g_r_w_r += rmse_g_r_w_r
                    total_rmse_u_r_w_r += rmse_u_r_w_r
                    mae_r = mean_absolute_error(pred_ratings_org, 5, 1)
                    mae_g_r_w_r = mean_absolute_error(g_r_w_r, 5, 1)
                    mae_u_r_w_r = mean_absolute_error(u_r_w_r, 5, 1)

                    total_mae_r += mae_r
                    total_mae_g_r_w_r += mae_g_r_w_r
                    total_mae_u_r_w_r += mae_u_r_w_r

                    #===========user and group label accuracy===========

                    total_group_attr_acc += self.compute_accuracy(opinion_labels, group_pred_labels)
                    total_user_attr_acc += self.compute_accuracy(opinion_labels, user_pred_labels)
                    all_gt.append(opinion_labels)  # gt_label: tensor
                    all_pred.append(group_pred_labels)




                else:
                    total_group_attr_acc = 0
                    total_user_attr_acc = 0
                    total_rmse_r = 0
                    total_mae_r = 0
                    total_mae_r_w_r = 0
                    total_rmse_r_w_r = 0



                # count += 1

            self.plot_simple_distribution(true_ratings, group_pred_ratings)
            all_gt = torch.cat(all_gt, dim=0)  # shape: [Total_samples, 7, 4]
            all_pred = torch.cat(all_pred, dim=0)  # shape: [Total_samples, 7, 4]
            group_metrics = self.evaluate_opinion_metrics(all_gt, all_pred)

            # avg_rating_loss = total_rating_loss / len(self.test_loader)
            # rmse = torch.sqrt(torch.tensor(avg_rating_loss))
            rating_RMSE = total_rmse / len(self.test_loader)
            rating_MAE = total_mae / len(self.test_loader)
            avg_mae = total_mae / len(self.test_loader)
            # new_rating_RMSE = total_rmse_r / len(self.test_loader)
            # new_rating_MAE = total_mae_r / len(self.test_loader)
            rating_with_group_reviews_RMSE = total_rmse_g_r_w_r / len(self.test_loader)
            rating_with_user_RMSE = total_rmse_u_r_w_r / len(self.test_loader)
            rating_with_group_reviews_MAE = total_mae_g_r_w_r / len(self.test_loader)
            rating_with_user_MAE = total_mae_u_r_w_r / len(self.test_loader)
            user_label_acc = total_user_attr_acc / len(self.test_loader)
            group_label_acc = total_group_attr_acc / len(self.test_loader)

        # eval_results = collector.get_result()alb

        test_results = {
            # "Recall@10": recall_at_10 / len(self.test_loader),
            # "Recall@20": recall_at_20 / len(self.test_loader),
            # "NDCG@10": ndcg_at_10 / len(self.test_loader),
            # "NDCG@20": ndcg_at_20 / len(self.test_loader),
            "only test rating RMSE:": rating_RMSE,
            "only test rating MAE:": rating_MAE,
            # "new rating RMSE:": new_rating_RMSE,
            # "new rating MAE:": new_rating_MAE,
            "rating_with_user_RMSE:": rating_with_user_RMSE,
            "rating_with_group_reviews_RMSE:": rating_with_group_reviews_RMSE,
            "rating_with_user_MAE:": rating_with_user_MAE,
            "rating_with_group_reviews_MAE:": rating_with_group_reviews_MAE,
            "Accuracy for user opinions labels": user_label_acc,
            "Accuracy for group opinions labels": group_label_acc
        }
        print("🎯 Final Test Results:", test_results)
        print("Lable test metrics", group_metrics)
        # print(f"🏆 Best Model Final Test Recall@10: {test_results['Recall@10']:.4f}")

        return rating_RMSE, test_results

        # test_results = collector.get_result()
        # print("🎯 Final Test Results:", test_results)

        # Final validation score for model selection
        # final_valid_score = calculate_valid_score(test_results, "Recall@10")
        # print(f"🏆 Best Model Final Test Recall@10: {final_valid_score:.4f}")

    def plot_loss_curves(self, train_losses, val_losses, save_path):
        """
        Plots and saves training and validation loss curves on the same figure.

        Args:
            train_losses (list): Training loss per epoch.
            val_losses (list): Validation loss per epoch.
            save_path (str): Path to save the plot.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
        plt.plot(epochs, val_losses, label='Validation Loss', marker='s', color='orange')

        plt.title("Training & Validation Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"📉 Loss curves saved to {save_path}")



    def evaluate_opinion_metrics(self, opinion_labels, group_pred_labels):
        """
        opinion_labels: torch.Tensor [B,7,4], GT one-hot
        group_pred_labels: torch.Tensor [B,7,4], predicted probabilities
        Returns: dict of precision, recall, f1, auc per attribute and overall
        """
        device = opinion_labels.device
        B, A, C = opinion_labels.shape

        # ===== Convert to numpy =====
        probs = group_pred_labels.detach().cpu().numpy()  # [B,7,4]

        # ===== Convert GT to indices =====
        gt_indices = opinion_labels.argmax(dim=-1).view(-1).cpu().numpy()  # [B*7]

        # ===== Predicted indices =====
        pred_indices = group_pred_labels.argmax(dim=-1).view(-1).cpu().numpy()  # [B*7]

        # ===== Mask invalid labels =====
        valid_mask = (opinion_labels.sum(dim=-1) > 0).view(-1).cpu().numpy()  # [B*7]

        metrics = {}
        all_prec, all_rec, all_f1, all_auc = [], [], [], []

        # ===== Overall (macro) metrics =====
        if valid_mask.sum() > 0:
            overall_prec = precision_score(gt_indices[valid_mask], pred_indices[valid_mask], average='macro',
                                           zero_division=0)
            overall_rec = recall_score(gt_indices[valid_mask], pred_indices[valid_mask], average='macro',
                                       zero_division=0)
            overall_f1 = f1_score(gt_indices[valid_mask], pred_indices[valid_mask], average='macro', zero_division=0)

            # ===== Overall AUC =====
        #     probs_flat = probs.reshape(-1, C)  # [B*7, 4]
        #     pos_probs = probs_flat[:, 1]  # example: positive class = index 1
        #     masked_probs = pos_probs[valid_mask]
        #     masked_gt_bin = (gt_indices[valid_mask] == 1).astype(int)  # binary gt for positive class
        #
        #     try:
        #         overall_auc = roc_auc_score(masked_gt_bin, masked_probs)
        #     except ValueError:
        #         overall_auc = 0.0
        #
        else:
            overall_prec = overall_rec = overall_f1 = 0.0

        metrics['overall_precision'] = overall_prec
        metrics['overall_recall'] = overall_rec
        metrics['overall_f1'] = overall_f1
        # metrics['overall_auc'] = overall_auc

        # ===== Per attribute metrics =====
        for attr in range(A):
            gt_attr = opinion_labels.argmax(dim=-1)[:, attr].cpu().numpy()  # [B]
            pred_attr = group_pred_labels.argmax(dim=-1)[:, attr].cpu().numpy()  # [B]
            valid_attr = (opinion_labels.sum(dim=-1)[:, attr] > 0).cpu().numpy()  # [B]

            if valid_attr.sum() > 0:
                p = precision_score(gt_attr[valid_attr], pred_attr[valid_attr], average='macro', zero_division=0)
                r = recall_score(gt_attr[valid_attr], pred_attr[valid_attr], average='macro', zero_division=0)
                f = f1_score(gt_attr[valid_attr], pred_attr[valid_attr], average='macro', zero_division=0)

                # ===== Per attribute AUC =====
                # probs_attr = probs[:, attr, :]  # [B, 4]
                # pos_probs_attr = probs_attr[:, 1]  # positive class = index 1
                # masked_probs_attr = pos_probs_attr[valid_attr]
                # masked_gt_attr_bin = (gt_attr[valid_attr] == 1).astype(int)
                #
                # try:
                #     auc = roc_auc_score(masked_gt_attr_bin, masked_probs_attr)
                # except ValueError:
                #     auc = 0.0

            else:
                p = r = f = auc = 0.0

            metrics[f'attr{attr}_precision'] = p
            metrics[f'attr{attr}_recall'] = r
            metrics[f'attr{attr}_f1'] = f
            # metrics[f'attr{attr}_auc'] = auc

            all_prec.append(p)
            all_rec.append(r)
            all_f1.append(f)
            # all_auc.append(auc)

        return metrics

    def compute_accuracy(self, true_labels, predicted_labels):
        """
        true_labels: [B, 7, 4] one-hot
        predicted_labels: [B, 7, 4] logits or probs
        """
        with torch.no_grad():
            # [B, 7] → bool，表示该位置是否为有效标签
            valid_mask = true_labels.sum(dim=-1) > 0  # [B, 7]

            # 获取预测类别：取最大概率的位置
            pred_classes = torch.argmax(predicted_labels, dim=-1)  # [B, 7]
            true_classes = torch.argmax(true_labels, dim=-1)  # [B, 7]

            # 只保留有效标签位置
            correct = (pred_classes == true_classes) & valid_mask  # [B, 7]
            num_correct = correct.sum().item()
            num_valid = valid_mask.sum().item()

            accuracy = num_correct / num_valid if num_valid > 0 else 0.0
            return accuracy

    from matplotlib.gridspec import GridSpec

    def plot_rating_distribution(self, g_r_w_r, save_path="rating_analysis.png"):
        """
        绘制真实评分(true_rating)和预测评分(pred_rating)的联合分布
        参数:
            g_r_w_r: [(真实评分, 预测评分)] 的列表
            save_path: 图片保存路径
        """
        # 解压数据
        true_ratings, pred_ratings = zip(*g_r_w_r)
        true_ratings = np.array(true_ratings)
        pred_ratings = np.array(pred_ratings)
        errors = true_ratings - pred_ratings

        # 创建画布
        plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, width_ratios=[2, 2, 1])

        # 1. 双变量分布图
        ax1 = plt.subplot(gs[0])
        sns.kdeplot(x=true_ratings, y=pred_ratings,
                    cmap="Blues", fill=True, ax=ax1)
        max_rating = max(np.max(true_ratings), np.max(pred_ratings))
        ax1.plot([0, max_rating], [0, max_rating], 'r--', alpha=0.5)  # 理想对角线
        ax1.set_title('True vs Predicted Rating Distribution')
        ax1.set_xlabel('True Rating')
        ax1.set_ylabel('Predicted Rating')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # 2. 误差分布图
        ax2 = plt.subplot(gs[1])
        sns.histplot(errors, bins=20, kde=True,
                     color='purple', ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Prediction Error Distribution\n(True - Predicted)')
        ax2.set_xlabel('Error')
        ax2.grid(True, linestyle='--', alpha=0.3)

        # 3. 统计指标
        ax3 = plt.subplot(gs[2])
        ax3.axis('off')
        metrics = [
            f"Total Samples: {len(true_ratings)}",
            f"True Avg: {np.mean(true_ratings):.2f} ± {np.std(true_ratings):.2f}",
            f"Pred Avg: {np.mean(pred_ratings):.2f} ± {np.std(pred_ratings):.2f}",
            f"MAE: {np.mean(np.abs(errors)):.2f}",
            f"RMSE: {np.sqrt(np.mean(errors ** 2)):.2f}",
            f"Correlation: {np.corrcoef(true_ratings, pred_ratings)[0, 1]:.2f}"
        ]
        ax3.text(0.1, 0.8, "\n".join(metrics),
                 fontfamily='monospace', fontsize=10,
                 bbox=dict(facecolor='whitesmoke', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rating analysis saved to {save_path}")

    def plot_rating_distribution(self, ratings, pred_ratings, output_dir="results"):
        """
        绘制真实评分和预测评分的分布对比图
        """
        plt.figure(figsize=(12, 6))

        # 真实评分分布
        plt.subplot(1, 2, 1)
        sns.histplot(ratings, bins=10, kde=True, color='blue')
        plt.title('Real Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')

        # 预测评分分布
        plt.subplot(1, 2, 2)
        sns.histplot(pred_ratings, bins=10, kde=True, color='orange')
        plt.title('Predicted Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(f"mmoe_rating_distribution.png")
        plt.close()

    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    def plot_simple_distribution(self, ratings, pred_ratings, save_path="mmoe_rating_distribution.png"):
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


    # def plot_simple_distribution(self, ratings, pred_ratings, save_path="mmoe_rating_distribution.png"):
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
    #     plt.savefig(f"mmoe_rating_distribution.png")
    #     plt.close()

    # def plot_simple_distribution(self, true_ratings, pred_ratings, save_path="rating_distribution.png"):
    #     """
    #     简化版评分分布可视化（仅直方图）
    #     参数:
    #         true_ratings: 真实评分列表/数组
    #         pred_ratings: 预测评分列表/数组
    #         save_path: 图片保存路径
    #     """
    #     plt.figure(figsize=(10, 5))
    #
    #     # 设置柱子边缘颜色
    #     edge_color = 'white'
    #
    #     # 真实评分分布
    #     plt.hist(true_ratings, bins=20, alpha=0.7, color='#3498db',
    #              edgecolor=edge_color, label='True Ratings', density=True)
    #
    #     # 预测评分分布
    #     plt.hist(pred_ratings, bins=20, alpha=0.7, color='#e74c3c',
    #              edgecolor=edge_color, label='Predicted Ratings', density=True)
    #
    #     # 图表装饰
    #     plt.title('Rating Distribution Comparison', pad=20)
    #     plt.xlabel('Rating Value')
    #     plt.ylabel('Normalized Frequency')
    #     plt.legend()
    #     plt.grid(axis='y', linestyle='--', alpha=0.4)
    #
    #     # 自动调整x轴范围
    #     min_val = min(np.min(true_ratings), np.min(pred_ratings))
    #     max_val = max(np.max(true_ratings), np.max(pred_ratings))
    #     plt.xlim(min_val - 0.5, max_val + 0.5)
    #
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=120)
    #     print(f"Distribution plot saved to {save_path}")

    # 使用示例
    if __name__ == "__main__":
        # 模拟数据（替换为您的真实数据）
        np.random.seed(42)
        true_rating = np.clip(np.random.normal(3.5, 1, 1000), 1, 5)
        pred_rating = true_rating + np.random.normal(0, 0.3, 1000)
        pred_rating = np.clip(pred_rating, 1, 5)

        # 绘制分布图
        plot_simple_distribution(true_rating, pred_rating)