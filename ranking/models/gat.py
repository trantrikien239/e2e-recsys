# Group project for CSE 6240: Web Search and Text Mining at Georgia Tech
# Author: Kien Tran (github.com/trantrikien239)
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, to_hetero
import torch.nn.functional as F

from interaction_func import DotProduct, MLP2LayersV1, MLP2LayersV2

class GAT2Layers(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads=1):
        super().__init__()
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        self.in_channels = hidden_channels
        self.out_channels = hidden_channels // num_heads
        self.conv1 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)
        self.conv2 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GAT3Layers(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads=1):
        super().__init__()
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        self.in_channels = hidden_channels
        self.out_channels = hidden_channels // num_heads
        self.num_heads = num_heads

        self.conv1 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)
        self.conv2 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)
        self.conv3 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
class GAT2LayersTFM(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads=1, dim_feedforward=512):
        super().__init__()
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        self.in_channels = hidden_channels
        self.out_channels = hidden_channels // num_heads
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.conv1 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)
        self.conv2 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)
        self.ffn1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, hidden_channels)
        )
        self.norm_ff1 = torch.nn.LayerNorm(hidden_channels)
        self.ffn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, hidden_channels)
        )
        self.norm_ff2 = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = x + self.ffn1(x) # Residual connection
        x = self.norm_ff1(x)
        x = self.conv2(x, edge_index)
        x = x + self.ffn2(x) # Residual connection
        x = self.norm_ff2(x)
        return x

    
class GAT3LayersTFM(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads=1, dim_feedforward=512):
        super().__init__()
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        self.in_channels = hidden_channels
        self.out_channels = hidden_channels // num_heads
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.conv1 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)
        self.conv2 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)
        self.conv3 = GATv2Conv(self.in_channels, self.out_channels, heads=num_heads, add_self_loops=False)
        self.ffn1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, hidden_channels)
        )
        self.norm_ff1 = torch.nn.LayerNorm(hidden_channels)
        self.ffn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, hidden_channels)
        )
        self.norm_ff2 = torch.nn.LayerNorm(hidden_channels)
        self.ffn3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, hidden_channels)
        )
        self.norm_ff3 = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = x + self.ffn1(x)
        x = self.norm_ff1(x)
        x = self.conv2(x, edge_index)
        x = x + self.ffn2(x)
        x = self.norm_ff2(x)
        x = self.conv3(x, edge_index)
        x = x + self.ffn3(x)
        x = self.norm_ff3(x)
        return x

    
class GATLinkPred(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 n_user, n_prod, 
                 user_feat_size, prod_feat_size, 
                 dataset_metadata,
                 n_gnn_layers=3,
                 num_heads=1,
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
            self.gnn = GAT3Layers(hidden_channels, num_heads=num_heads)
        elif n_gnn_layers == 2:
            self.gnn = GAT2Layers(hidden_channels, num_heads=num_heads)
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
    
    
class GATLinkPredNoEmb(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 user_feat_size, prod_feat_size, 
                 dataset_metadata,
                 n_gnn_layers=3,
                 num_heads=1,
                 interaction_func="dot_product"):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.user_lin = torch.nn.Linear(user_feat_size, hidden_channels)
        self.prod_lin = torch.nn.Linear(prod_feat_size, hidden_channels)

        # Instantiate homogeneous GNN:
        if n_gnn_layers == 3:
            self.gnn = GAT3Layers(hidden_channels, num_heads=num_heads)
        elif n_gnn_layers == 2:
            self.gnn = GAT2Layers(hidden_channels, num_heads=num_heads)
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