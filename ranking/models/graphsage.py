# Group project for CSE 6240: Web Search and Text Mining at Georgia Tech
# Author: Kien Tran (github.com/trantrikien239)
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

from .interaction_func import DotProduct, MLP2LayersV1, MLP2LayersV2

class GraphSage2Layers(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphSage3Layers(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
class GraphSageLinkPred(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 n_user, n_prod, 
                 user_feat_size, prod_feat_size, 
                 dataset_metadata,
                 n_gnn_layers=3,
                 interaction_func="dot_product"):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.user_lin = torch.nn.Linear(user_feat_size, hidden_channels)
        self.prod_lin = torch.nn.Linear(prod_feat_size, hidden_channels)
        self.user_emb = torch.nn.Embedding(n_user, hidden_channels)
        self.prod_emb = torch.nn.Embedding(n_prod, hidden_channels)

        # Instantiate homogeneous GNN:
        if n_gnn_layers == 3:
            self.gnn = GraphSage3Layers(hidden_channels)
        elif n_gnn_layers == 2:
            self.gnn = GraphSage2Layers(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=dataset_metadata)

        # Instantiate interaction function:
        if interaction_func == "dot_product":
            self.classifier = DotProduct()
        elif interaction_func == "mlp2layers_v1":
            self.classifier = MLP2LayersV1(hidden_channels)
        elif interaction_func == "mlp2layers_v2":
            self.classifier = MLP2LayersV2(hidden_channels)

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_lin(data["user"].x) + self.user_emb(data["user"].node_id),
          "prod": self.prod_lin(data["prod"].x) + self.prod_emb(data["prod"].node_id),
        } 

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["prod"],
            data["user", "buy", "prod"].edge_label_index,
        )
        return pred
    
class GraphSageLinkPredNoEmb(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 user_feat_size, prod_feat_size, 
                 dataset_metadata,
                 n_gnn_layers=3,
                 interaction_func="dot_product"):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.user_lin = torch.nn.Linear(user_feat_size, hidden_channels)
        self.prod_lin = torch.nn.Linear(prod_feat_size, hidden_channels)
        # self.user_emb = torch.nn.Embedding(n_user, hidden_channels)
        # self.prod_emb = torch.nn.Embedding(n_prod, hidden_channels)

        # Instantiate homogeneous GNN:
        if n_gnn_layers == 3:
            self.gnn = GraphSage3Layers(hidden_channels)
        elif n_gnn_layers == 2:
            self.gnn = GraphSage2Layers(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=dataset_metadata)

        # Instantiate interaction function:
        if interaction_func == "dot_product":
            self.classifier = DotProduct()
        elif interaction_func == "mlp2layers_v1":
            self.classifier = MLP2LayersV1(hidden_channels)
        elif interaction_func == "mlp2layers_v2":
            self.classifier = MLP2LayersV2(hidden_channels)

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_lin(data["user"].x),
          "prod": self.prod_lin(data["prod"].x),
        } 

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["prod"],
            data["user", "buy", "prod"].edge_label_index,
        )
        return pred
    
    def get_user_prod_embeddings(self, data: HeteroData):
        x_dict = {
          "user": self.user_lin(data["user"].x),
          "prod": self.prod_lin(data["prod"].x),
        } 

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        edge_feat_user = x_dict["user"][data["user", "buy", "prod"].edge_label_index[0]]
        edge_feat_prod = x_dict["prod"][data["user", "buy", "prod"].edge_label_index[1]]
        return edge_feat_user, edge_feat_prod