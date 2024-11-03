FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt --no-cache-dir

ENV EDGE_PATH="data/val_edges.parquet"
ENV MS_ALS_PATH="data/models/als_latest.npz"

EXPOSE 8000

CMD ["python", "main.py"]