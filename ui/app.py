import os
import pandas as pd
from functools import reduce


import streamlit as st
from utils import candidate_gen_rw, candidate_gen_pop, candidate_gen_mf,\
    ranking

RETRIEVAL_API_URL = os.environ.get('RETRIEVAL_API_URL', 'http://localhost:8000')
RANKING_API_URL = os.environ.get('RANKING_API_URL', 'http://localhost:8001')
done_retrieval = False

st.set_page_config(layout='wide')

st.title('Demo: Stages of recommendation')
st.write("""This is a demo of a recommendation system. The system is based on a graph of user-product interactions. 
A typical recommendation system has the following stages:

1. Retrieval (aka candidate generation): This stage generates a list of hundreds of candidate items for a user.
2. Ranking: This stage ranks the candidate items based on user preferences using more complex models.

The system is built using FastAPI, Streamlit and deployed using Docker and Kubernetes.""")

st.header('Stage 1: Retrieval')
# 2 columns of user input
col1, col2, col3 = st.columns(3)

# Get user id
user_id = col1.number_input('Enter user id:', min_value=0, max_value=300_000)
n_candidates = col2.number_input('Number of candidates:', min_value=1, max_value=500, value=10)
# Submit button
submit = col3.button('Get candidates')

conf_rw, conf_mf, conf_pop = st.columns(3)

conf_rw.header('Random walk')
crw1, crw2 = conf_rw.columns(2)
rw_n_hops = crw1.number_input('Number of hops:', min_value=1, max_value=10, value=2)
rw_min_cnt = crw2.number_input('Minimum count:', min_value=1, max_value=100, value=2)

conf_mf.header('Matrix factorization')

conf_pop.header('Popularity')

res_rw, res_mf, res_pop = st.columns(3)

st.header('Merged list of candidates')

if submit:
    candidates, info = candidate_gen_rw(RETRIEVAL_API_URL, user_id, n_candidates, rw_n_hops, rw_min_cnt)
    if candidates:
        conf_rw.write(info)
        res_rw.write('Top candidates:')
        df_candidates = pd.DataFrame(candidates, columns=['product_id', 'random_walk_cnt'])
        res_rw.dataframe(df_candidates)
    
    pop_candidates = candidate_gen_pop(RETRIEVAL_API_URL, user_id, n_candidates)
    if pop_candidates:
        res_pop.write('Top candidates:')
        df_pop_candidates = pd.DataFrame(pop_candidates, columns=['product_id', 'popularity'])
        res_pop.dataframe(df_pop_candidates)

    mf_candidates = candidate_gen_mf(RETRIEVAL_API_URL, user_id, n_candidates)
    if mf_candidates:
        res_mf.write('Top candidates:')
        df_mf_candidates = pd.DataFrame(mf_candidates, columns=['product_id', 'mf_score'])
        res_mf.dataframe(df_mf_candidates)    

    df_merged = reduce(
        lambda  left,right: pd.merge(left,right,on=['product_id'], how='outer'), 
        [df_candidates, df_pop_candidates, df_mf_candidates])

    # df_merged = pd.concat(
    #     [df_candidates, df_pop_candidates, df_mf_candidates], 
    #     axis=1, join='outer', keys='product_id')
    st.dataframe(df_merged)
    st.code(df_merged['product_id'].to_list())
    done_retrieval = True

st.header('Stage 2: Ranking')
st.write("Ranking is done using a GraphSage model. The model can leverage structural information of the interaction graph and can use sparse and dense features of the users and products. The model is then used to rank the candidate items generated in the retrieval stage.")

if done_retrieval:
    ranked = ranking(RANKING_API_URL, user_id, df_merged['product_id'].to_list())
    st.write('Ranked candidates:')
    df_ranked_ = pd.DataFrame(ranked, columns=['product_id', 'ranking_score'])
    df_ranked = pd.merge(df_merged, df_ranked_, on='product_id', how='inner')
    df_ranked.sort_values('ranking_score', ascending=False, inplace=True)
    df_ranked.reset_index(drop=True, inplace=True)
    st.dataframe(df_ranked)
