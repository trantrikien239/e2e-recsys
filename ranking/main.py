import os

import numpy as np
import pandas as pd
import torch

import uvicorn

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from torch_geometric.loader import LinkNeighborLoader

from models.utils import prep_dataset
from models.graphsage import GraphSage2Layers, GraphSage3Layers, GraphSageLinkPred,\
    GraphSageLinkPredNoEmb


GRAPHSAGE_MODEL_PATH = os.environ.get(
    "GRAPHSAGE_MODEL_PATH", 
    "data/models/gs_adv_noemb_2gnn.pkl")
NODE2VEC_EMB_PATH = os.environ.get(
    "NODE2VEC_EMB_PATH",
    "data/node2vec_128_v2.pkl")
TRAIN_EDGE_FILE = os.environ.get(
    "TRAIN_EDGE_FILE",
    "../retrieval/data/train_edges.parquet")
VAL_EDGE_FILE = os.environ.get(
    "VAL_EDGE_FILE",
    "../retrieval/data/val_edges.parquet")



class AppState:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prod_feat = np.load('data/products.npy')
        user_feat = np.load('data/train_user_features_norm.npy')
        with open(NODE2VEC_EMB_PATH, "rb") as f:
            z = torch.load(f, map_location=torch.device('cpu'))
        
        train_edges = pd.read_parquet(TRAIN_EDGE_FILE)
        enc_user_id = train_edges[["user_id"]].drop_duplicates()\
            .sort_values("user_id").reset_index(drop=True).reset_index()\
                .rename(columns={"index": "enc_user_id"})
        self.enc_user_id_dict = dict(zip(enc_user_id.user_id, enc_user_id.enc_user_id))
        train_edges["enc_user_id"] = train_edges["user_id"].map(self.enc_user_id_dict)

        map_uid = train_edges[["user_id", "enc_user_id"]].drop_duplicates()\
            .sort_values("enc_user_id").reset_index(drop=True)
        map_uid["fake_uid"] = map_uid["user_id"] + 50000

        n2v_user_emb = z[map_uid["fake_uid"],:].detach().cpu().numpy()
        n_prod = prod_feat.shape[0]
        n2v_prod_emb = z[:n_prod,:].detach().cpu().numpy()
        
        self.prod_features = np.hstack([n2v_prod_emb, prod_feat])
        self.user_features = np.hstack([n2v_user_emb, user_feat])

        
        

        self.val_edges = pd.read_parquet(VAL_EDGE_FILE)
        self.val_edges["enc_user_id"] = self.val_edges["user_id"].map(self.enc_user_id_dict)


        self.n_user, self.user_feat_size = self.user_features.shape
        self.n_prod, self.prod_feat_size = self.prod_features.shape
        self.dl_meta_data = (['user', 'prod'], [('user', 'buy', 'prod'), ('prod', 'rev_buy', 'user')])
        self.graphsage_model = GraphSageLinkPredNoEmb(hidden_channels=256,
                 user_feat_size=self.user_feat_size,
                 prod_feat_size=self.prod_feat_size,
                 dataset_metadata=self.dl_meta_data,
                 n_gnn_layers=2,
                 interaction_func="dot_product")
        with open("data/models/gs_adv_noemb_2gnn.pkl", "rb") as f:
            self.graphsage_model = torch.load(f, map_location=self.device)
        
        del train_edges
        
@asynccontextmanager
async def lifespan(app):
    app.state = AppState()
    yield
    del app.state

app = FastAPI(lifespan=lifespan)

@app.post("/rank_graphsage/")
async def rank_graphsage(
    user_id:int, 
    prod_id_candidates:list[int]):
    try:
        val_labels_to_pred = pd.DataFrame({
            "user_id": [user_id] * len(prod_id_candidates),
            "product_id": prod_id_candidates,
            "label": [0] * len(prod_id_candidates),
        })
        val_labels_to_pred['enc_user_id'] = val_labels_to_pred["user_id"].map(
            app.state.enc_user_id_dict)
        val_data = prep_dataset(app.state.val_edges,
                        val_labels_to_pred,
                        app.state.user_features, 
                        app.state.prod_features)
        # Define the validation seed edges:
        val_loader = LinkNeighborLoader(
            data=val_data,
            # num_neighbors={("user", "buy", "prod"): [100,100],
            #             ("prod", "rev_buy", "user"): [50, 50]},
            num_neighbors=[10,20,10],
            edge_label_index=(("user", "buy", "prod"),
                            val_data["user", "buy", "prod"].edge_label_index),
            batch_size=200,
            shuffle=False,
        )

        preds = []
        for sampled_data in val_loader:
            with torch.no_grad():
                sampled_data.to(app.state.device)
                preds.append(app.state.graphsage_model(sampled_data))

        pred = torch.cat(preds, dim=0).cpu().numpy()
        val_labels_to_pred['pred'] = pred
        val_labels_to_pred.sort_values("pred", ascending=False, inplace=True)
        out = []
        for p_, s_ in zip(val_labels_to_pred["product_id"], val_labels_to_pred["pred"]):
            out.append([int(p_), float(s_)])
        return {"ranked_candidates": out}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)