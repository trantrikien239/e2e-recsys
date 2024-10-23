import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

TRAIN_EDGE_FILE = 'data/train_edges.parquet'
TRAINED_MODEL_FILE = 'data/models/als_latest.npz'

model = AlternatingLeastSquares(
        factors=100, regularization=0.01, iterations=20, 
        use_gpu=False, calculate_training_loss=True)

if __name__ == '__main__':
    # Load the data
    train_edges = pd.read_parquet(TRAIN_EDGE_FILE)

    user_item_csr = csr_matrix(
        (
            np.ones(train_edges.shape[0]),
            (
                train_edges['user_id'].values,
                train_edges['product_id'].values
            )
        ),
        shape=(
            train_edges['user_id'].max() + 1,
            train_edges['product_id'].max() + 1
        )
    )
    
    model.fit(user_item_csr)

    model.save(TRAINED_MODEL_FILE)

    print("Model trained and saved successfully!")
    
