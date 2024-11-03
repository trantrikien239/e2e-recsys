FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# ENV RETRIEVAL_API_URL="http://retrieval:8000"
# ENV RANKING_API_URL="http://ranking:8001"

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]