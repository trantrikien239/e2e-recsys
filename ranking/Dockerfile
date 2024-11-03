FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install  --no-cache-dir -r requirements.txt

# Install PyTorch Geometric dependencies with --find-links
RUN pip install \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cpu.html \
    --no-build-isolation

ENV GRAPHSAGE_MODEL_PATH="data/models/gs_adv_noemb_2gnn.pkl"
ENV NODE2VEC_EMB_PATH="data/node2vec_128_v2.pkl"
ENV TRAIN_EDGE_FILE="data/train_edges.parquet"
ENV VAL_EDGE_FILE="data/val_edges.parquet"

EXPOSE 8001
CMD ["python", "main.py"]