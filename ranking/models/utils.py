# Group project for CSE 6240: Web Search and Text Mining at Georgia Tech
# Author: Kien Tran (github.com/trantrikien239)
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm

import torch.nn.functional as F



def prep_dataset(df_edges, df_labels, user_features, prod_features):
    edge_index = torch.tensor(df_edges[["enc_user_id", "product_id"]].values, 
                                dtype=torch.long).t().contiguous()
    data = HeteroData()

    # Save node indices:
    data["user"].node_id = torch.arange(user_features.shape[0])
    data["prod"].node_id = torch.arange(prod_features.shape[0])


    # Add the node features and edge indices:
    data["user"].x = torch.tensor(user_features, dtype=torch.float32)
    data["prod"].x = torch.tensor(prod_features, dtype=torch.float32)
    data["user", "buy", "prod"].edge_index = edge_index

    data = T.ToUndirected()(data)
    data["user", "buy", "prod"].edge_label_index = torch.tensor(
        df_labels[["enc_user_id", "product_id"]].values, dtype=torch.long).t().contiguous()
    data["user", "buy", "prod"].edge_label = torch.tensor(
        df_labels["label"].values, dtype=torch.float32).t().contiguous()
    return data


def eval_auc(model, val_loader, device):
    preds = []
    ground_truths = []
    for sampled_data in tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["user", "buy", "prod"].edge_label)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    return auc, ground_truth, pred

def train(model, train_loader, device, epochs=8):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = total_examples = 0
        for sampled_data in tqdm(train_loader):
            optimizer.zero_grad()

            sampled_data.to(device)
            pred = model(sampled_data)

            ground_truth = sampled_data["user", "buy", "prod"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    return model

def train_early_stop(model, train_loader, val_loader, device, lr=0.001, epochs=10, return_stats=False):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc = 0
    best_model = None
    stats = [["epoch", "train_loss", "val_auc", "val_precision", "val_recall", "val_f1"]]

    for epoch in range(epochs):
        total_loss = total_examples = 0
        for sampled_data in tqdm(train_loader):
            optimizer.zero_grad()

            sampled_data.to(device)
            pred = model(sampled_data)

            ground_truth = sampled_data["user", "buy", "prod"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print("Evaluating on the lite validation set...")
        auc_val, gt_val, pred_val = eval_auc(model, val_loader, device)
        pred_proba_val = torch.sigmoid(torch.tensor(pred_val)).cpu().numpy()
        pred_binary_val = (pred_proba_val > 0.5).astype(int)
        precision_val = precision_score(gt_val, pred_binary_val)
        recall_val = recall_score(gt_val, pred_binary_val)
        f1_val = f1_score(gt_val, pred_binary_val)
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}, AUC: {auc_val:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1: {f1_val:.4f}")

        if auc_val > best_auc:
            best_auc = auc_val
            best_model = model.state_dict()
            stats.append([epoch, total_loss / total_examples, auc_val, precision_val, recall_val, f1_val])
        
        if auc_val < best_auc and epoch >= 2:
            print(f"Early stopping at epoch {epoch}")
            break
    if best_model is not None:
        model.load_state_dict(best_model)
    if return_stats:
        df_stats = pd.DataFrame(stats[1:], columns=stats[0])
        return model, df_stats
    else:
        return model