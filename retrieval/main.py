import os

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager

from random_walk import candidate_gen_hop
from matrix_factorization import model as als_model

EDGE_PATH = os.environ.get('EDGE_PATH', 'data/val_edges.parquet')
MF_ALS_PATH = os.environ.get('MF_ALS_PATH', 'data/models/als_latest.npz')

class AppState:
    def __init__(self):
        edge_df = pd.read_parquet(EDGE_PATH)
        edge_df['org_user_id'] = edge_df['user_id']
        edge_df['user_id'] = edge_df['user_id'] + 50_000

        self.G = nx.Graph()
        self.G.add_edges_from(edge_df[['user_id', 'product_id']].values)
        self.item_popularity = dict(self.G.degree())
        self.als_model = als_model.load(MF_ALS_PATH)
        self.user_item_csr = csr_matrix(
            (
                np.ones(edge_df.shape[0]),
                (
                    edge_df['org_user_id'].values,
                    edge_df['product_id'].values
                )
            ),
            shape=(
                edge_df['org_user_id'].max() + 1,
                edge_df['product_id'].max() + 1
            )
        )
        del edge_df

@asynccontextmanager
async def lifespan(app):
    app.state = AppState()
    yield
    del app.state


app = FastAPI(lifespan=lifespan)

@app.get("/randomwalk_candidates/")
async def randomwalk_candidates(
    user_id:int, 
    n_candidates:int=100, 
    n_hop:int=2, 
    min_cnt:int=2):
    try:
        uid_ = user_id + 50_000
        if uid_ not in app.state.G.nodes:
            raise HTTPException(status_code=404, detail="User not found")
        top_candidates, info_ = candidate_gen_hop(
            app.state.G, uid_, n_hop, n_candidates, min_cnt=min_cnt)
        
        return {'candidates': [sorted(top_candidates, key=lambda x: -x[1])],
                'info': [info_]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/randomwalk_candidates_batch/")
async def randomwalk_candidates_batch(
    list_user_id:list[int], 
    n_candidates:int=100, 
    n_hop:int=2, 
    min_cnt:int=2):
    try:
        out = []
        info = []
        for user_id in list_user_id:
            uid_ = user_id + 50_000
            if uid_ not in app.state.G.nodes:
                out.append([-1] * n_candidates)
                continue
            top_candidates, info_ = candidate_gen_hop(
                app.state.G, uid_, n_hop, n_candidates, min_cnt=min_cnt)
            out.append(sorted(top_candidates, key=lambda x: -x[1]))
            info.append(info_)
        return {'candidates': out, 'info': info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/popularity_candidates/")
async def popularity_candidates(user_id:int, n_candidates:int=100):
    try:
        uid_ = user_id + 50_000
        old_items = set(app.state.G.neighbors(uid_))
        top_candidates = sorted(app.state.item_popularity.items(), key=lambda x: -x[1])
        out = []
        for item, cnt in top_candidates:
            if item not in old_items:
                out.append([int(item), cnt])
            if len(out) >= n_candidates:
                break
        return {'candidates': [out]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mf_candidates/")
async def mf_candidates(user_id:int, n_candidates:int=100):
    try:
        uid_ = user_id
        ids, scores = app.state.als_model.recommend(
            uid_, app.state.user_item_csr[[uid_]], N=n_candidates,
            filter_already_liked_items=True, recalculate_user=True)
        out = []

        for item, score in zip(ids, scores):
            out.append([int(item), float(score)])
        return {'candidates': [out]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)