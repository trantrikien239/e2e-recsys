# Note: G must be an undirected, bipartite networkx graph
import random
from collections import defaultdict


def random_neigh(G, node):
    return random.choice(list(G.neighbors(node)))

def random_hop(G, start_node, n_hops):
    hop = [random_neigh(G, start_node)]
    for _ in range(n_hops):
        u_ = random_neigh(G, hop[-1])
        v_ = random_neigh(G, u_)
        hop.append(v_)
    return hop

def candidate_gen_hop(G, start_user_node, n_hops, n_candidates, 
                      max_walks=10_000, min_cnt=10):
    item_cnt = defaultdict(int)
    picked_items = set()
    old_items = set(G.neighbors(start_user_node))
    n_walks = 0
    real_walks = 0
    while n_walks < max_walks:
        hop = random_hop(G, start_user_node, n_hops)
        for node in hop:
            if node in old_items:
                continue
            item_cnt[node] += 1
            if item_cnt[node] >= min_cnt and node not in picked_items:
                picked_items.add(node)
            if len(picked_items) >= n_candidates:
                n_walks = max_walks
                real_walks += 1
                break
        n_walks += 1
        real_walks += 1
    return [[int(i_),item_cnt[i_]] for i_ in picked_items], {"n_walks": real_walks, "n_items_saw": len(item_cnt)}
