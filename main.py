import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import faiss
import pickle
import time
from rouge_score import rouge_scorer
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from sentence_transformers import SentenceTransformer

import matplotlib
matplotlib.use('Agg')  # Prevents GUI errors

# Initialize FastAPI
app = FastAPI()

# --------- SQLite Setup ---------
DB_CONN = sqlite3.connect("rag_db.sqlite", check_same_thread=False)
CURSOR = DB_CONN.cursor()

# Create tables for embeddings and logs
CURSOR.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    vector BLOB NOT NULL
)
""")

CURSOR.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    process_time REAL NOT NULL,
    retrieval_time REAL NOT NULL,
    rouge_score REAL NOT NULL,
    bert_score REAL NOT NULL
)
""")
DB_CONN.commit()

# --------- Load and Preprocess Data ---------
data = pd.read_csv('hotel_bookings.csv')
data.dropna(inplace=True)

# Add text column for FAISS search
data["text"] = data.apply(
    lambda row: f"Booking at {row['hotel']} on {row['arrival_date_year']}-{row['arrival_date_month']}. "
                f"Price: {row['adr']} per night. Canceled: {row['is_canceled']}", axis=1
)

# Encode data using SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(data["text"].tolist(), convert_to_numpy=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, "index.faiss")

# Store embeddings in SQLite
for text, vector in zip(data["text"], embeddings):
    vector_blob = pickle.dumps(vector)
    CURSOR.execute("INSERT INTO embeddings (text, vector) VALUES (?, ?)", (text, vector_blob))

DB_CONN.commit()

# --------- API Endpoints ---------

class QueryRequest(BaseModel):
    query: str

@app.post("/analytics")
def analytics():
    try:
        data_noncanc = data[data['is_canceled'] == 0].copy()
        data_noncanc["revenue"] = data_noncanc["adr"] * (data_noncanc["stays_in_weekend_nights"] + data_noncanc["stays_in_week_nights"])
        data_noncanc["revenue"] = data_noncanc["revenue"].astype(int)
        data_noncanc["date"] = pd.to_datetime(data_noncanc["arrival_date_year"].astype(str) + "-" + data_noncanc["arrival_date_month"] + "-" + data_noncanc["arrival_date_day_of_month"].astype(str))
        monthly_revenue = data_noncanc.groupby("date")["revenue"].sum().reset_index()

        total_revenue = data[data["is_canceled"] == 0]["adr"].sum()

        # Plot Monthly Revenue
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=monthly_revenue, x="date", y="revenue", markers="o", linewidth=2)
        plt.title("Monthly Revenue")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.grid(True)
        plt.savefig("monthly_revenue.png")
        plt.close()

        # Cancellation Rate
        total_bookings = data.shape[0]
        cancellations = data["is_canceled"].sum()
        cancellation_rate = (cancellations / total_bookings) * 100

        plt.figure(figsize=(8, 5))
        sns.histplot(data["is_canceled"], bins=2)
        plt.title(f"Cancellation Rate: {cancellation_rate:.2f}%")
        plt.savefig("cancellation_rate.png")
        plt.close()

        # Country-wise Booking Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data["country"])
        plt.title("Country-wise Booking Distribution")
        plt.xticks(rotation=45)
        plt.savefig("country_distribution.png")
        plt.close()

        # Booking Lead Time Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data_noncanc["lead_time"], bins=50, kde=True, color="skyblue")
        plt.title("Booking Lead Time Distribution")
        plt.xlabel("Lead Time (Days)")
        plt.ylabel("Number of Bookings")
        plt.grid(True)
        plt.savefig("lead_time_distribution.png")
        plt.close()

        # Guest Distribution Pie Chart
        Family = {
            "Adults": data_noncanc["adults"].sum(),
            "Children": data_noncanc["children"].sum(),
            "Babies": data_noncanc["babies"].sum()
        }

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(Family.values(), labels=Family.keys(), autopct='%1.1f%%', 
               colors=['lightblue', 'orange', 'green'], startangle=140, 
               wedgeprops={'edgecolor': 'black'})
        plt.title("Guest Distribution (Adults, Children, Babies)")
        plt.savefig("guest_distribution.png")
        plt.close()

        return JSONResponse(content={
            "total_bookings": total_bookings,
            "cancellation_rate": f"{cancellation_rate:.2f}%",
            "total_revenue": total_revenue,
            "plots": {
                "monthly_revenue": "/analytics_images/monthly_revenue.png",
                "cancellation_rate": "/analytics_images/cancellation_rate.png",
                "country_distribution": "/analytics_images/country_distribution.png",
                "guest_distribution": "/analytics_images/guest_distribution.png",
                "lead_time_distribution": "/analytics_images/lead_time_distribution.png"
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def QNA(request: QueryRequest):
    try:
        # Load FAISS index
        index = faiss.read_index("index.faiss")

        # Load text generation model
        global pipe
        pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", trust_remote_code=True)

        # Get query embedding
        query_embedding = model.encode(request.query, convert_to_numpy=True)

        # Retrieve similar results
        D, I = index.search(query_embedding.reshape(1, -1), 5)
        retrieved_texts = []

        for i in I[0]:
            CURSOR.execute("SELECT text FROM embeddings WHERE id=?", (i+1,))
            result = CURSOR.fetchone()
            if result:
                retrieved_texts.append(result[0])

        # Construct context and prompt
        context = " ".join(retrieved_texts)
        prompt = f"Context: {context}\nQuestion: {request.query}\nAnswer:"

        # Generate answer
        response = pipe(prompt, max_length=100, do_sample=True)[0]["generated_text"]

        return {"answer": response.strip()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --------- Middleware to Log Queries ---------
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()
    
    # Get request data
    request_body = await request.body()
    request_data = request_body.decode("utf-8")

    # Capture response
    response = await call_next(request)
    process_time = time.time() - start_time

    # Extract query from /ask endpoint
    if request.url.path == "/ask":
        query = eval(request_data).get("query", "Unknown query")
        
        # Measure FAISS retrieval time
        faiss_start = time.time()
        query_embedding = model.encode(query, convert_to_numpy=True)
        D, I = index.search(query_embedding.reshape(1, -1), 5)
        retrieval_time = time.time() - faiss_start

        # Get retrieved text
        retrieved_texts = []
        for i in I[0]:
            CURSOR.execute("SELECT text FROM embeddings WHERE id=?", (i+1,))
            result = CURSOR.fetchone()
            if result:
                retrieved_texts.append(result[0])

        # Generate response
        context = " ".join(retrieved_texts)
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        response_data = pipe(prompt, max_length=100, do_sample=True)[0]["generated_text"]

        # Compute ROUGE score
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        rouge_score = scorer.score(query, response_data)["rouge1"].fmeasure

        # Compute BERT similarity score
        query_vector = bert_model.encode(query, convert_to_numpy=True)
        response_vector = bert_model.encode(response_data, convert_to_numpy=True)
        bert_score = np.dot(query_vector, response_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(response_vector))

        # Store logs in SQLite
        CURSOR.execute("""
            INSERT INTO logs (query, response, process_time, retrieval_time, rouge_score, bert_score) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query, response_data, process_time, retrieval_time, rouge_score, bert_score))
        DB_CONN.commit()

    return response
@app.get("/health")
def health_check():
    return {"status": "OK", "database": "Connected", "faiss": "Loaded"}


# --------- Start FastAPI Server ---------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
