import requests
import streamlit as st
import json

@st.cache_data
def candidate_gen_rw(API_URL, user_id, n_candidates, n_hop=2, min_cnt=2):
    response = requests.get(f'{API_URL}/randomwalk_candidates/', params={'user_id': user_id, 'n_candidates': n_candidates, 'n_hop': n_hop, 'min_cnt': min_cnt})
    if response.status_code == 200:
        candidates = response.json()['candidates'][0]
        info = response.json()['info'][0]
        return candidates, info
    else:
        st.write(f'Error: {response.text}')
        return None, None
    
@st.cache_data
def candidate_gen_pop(API_URL, user_id, n_candidates):
    response = requests.get(f'{API_URL}/popularity_candidates/', params={'user_id': user_id, 'n_candidates': n_candidates})
    if response.status_code == 200:
        candidates = response.json()['candidates'][0]
        return candidates
    else:
        st.write(f'Error: {response.text}')
        return None
    
@st.cache_data
def candidate_gen_mf(API_URL, user_id, n_candidates):
    response = requests.get(f'{API_URL}/mf_candidates/', params={'user_id': user_id, 'n_candidates': n_candidates})
    if response.status_code == 200:
        candidates = response.json()['candidates'][0]
        return candidates
    else:
        st.write(f'Error: {response.text}')
        return None
    
@st.cache_data
def ranking(API_URL, user_id, prod_id_candidates):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    response = requests.post(
        url=f'{API_URL}/rank_graphsage/', 
        headers=headers,
        params={'user_id': user_id},
        data=json.dumps(prod_id_candidates))
    if response.status_code == 200:
        return response.json()['ranked_candidates']
    else:
        st.write(f'Error: {response.text}')
        return None