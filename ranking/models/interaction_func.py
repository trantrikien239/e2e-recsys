# Group project for CSE 6240: Web Search and Text Mining at Georgia Tech
# Author: Kien Tran (github.com/trantrikien239)
import torch
from torch import Tensor
import torch.nn.functional as F


class MLP2LayersV1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lin_cls_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin_cls_2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x_user: Tensor, x_prod: Tensor, edge_label_index: Tensor
                ) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_prod = x_prod[edge_label_index[1]]
        out = edge_feat_user * edge_feat_prod
        out = F.relu(self.lin_cls_1(out))
        out = self.lin_cls_2(out)

        # Apply dot-product to get a prediction per supervision edge:
        return out.squeeze()
    
class MLP2LayersV2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lin_cls_1 = torch.nn.Linear(input_dim * 2, hidden_dim)
        self.lin_cls_2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x_user: Tensor, x_prod: Tensor, edge_label_index: Tensor
                ) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_prod = x_prod[edge_label_index[1]]

        out = torch.cat([edge_feat_user, edge_feat_prod], dim=-1)
        out = F.relu(self.lin_cls_1(out))
        out = self.lin_cls_2(out)

        # Apply dot-product to get a prediction per supervision edge:
        return out.squeeze()
    
class DotProduct(torch.nn.Module):
    def forward(self, x_user: Tensor, x_prod: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_prod = x_prod[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_prod).sum(dim=-1)