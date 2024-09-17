from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import requests
import re

app = FastAPI()

# Hugging Face API information
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
headers = {"Authorization": f"Bearer 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"}

# Directory to save uploaded files (optional)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to query Hugging Face API for sentiment analysis
def query_huggingface_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Function to split text by conversation
def split_transcript_by_conversation(transcript, max_length=1500):
    pattern = r'(\[Sales Agent.*?\]|\[Customer.*?\])'
    segments = re.split(pattern, transcript)
    combined_segments = []
    for i in range(1, len(segments), 2):
        combined_segments.append(segments[i] + segments[i + 1])

    chunks = []
    current_chunk = ""
    for segment in combined_segments:
        if len(current_chunk) + len(segment) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = segment
        else:
            current_chunk += segment + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to extract top sentiment
def extract_top_sentiment(result):
    # Log the result for inspection
    print("Received result from API:", result)
    
    if isinstance(result, list) and len(result) > 0:
        # Check if the first item is also a list, then unwrap it
        if isinstance(result[0], list) and len(result[0]) > 0:
            result = result[0]
        
        # Find the top result by score
        if isinstance(result[0], dict):
            top_result = max(result, key=lambda x: x['score'])
            return {"label": top_result['label'], "score": top_result['score']}
        else:
            return {"label": "UNKNOWN", "score": 0}
    return {}

# FastAPI route for file upload
@app.post("/analyze")
async def analyze_sentiment(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, 'wb') as f:
        f.write(await file.read())

    with open(filepath, 'r', encoding='utf-8') as f:
        transcript = f.read()

    chunks = split_transcript_by_conversation(transcript)
    results = []
    positive_chunks = 0
    negative_chunks = 0
    neutral_chunks = 0

    
    for chunk in chunks:
        chunk_result = query_huggingface_api({"inputs": chunk})
        top_sentiment = extract_top_sentiment(chunk_result)
        
        # Check the top label and increment respective counters
        if top_sentiment["label"] == "positive":
            positive_chunks += 1
        elif top_sentiment["label"] == "negative":
            negative_chunks += 1
        elif top_sentiment["label"] == "neutral":
            neutral_chunks += 1
        
        results.append(top_sentiment)

    # Final Aggregated Result based on chunk counts
    if positive_chunks > negative_chunks and positive_chunks > neutral_chunks:
        overall_sentiment = "POSITIVE"
    elif negative_chunks > positive_chunks and negative_chunks > neutral_chunks:
        overall_sentiment = "NEGATIVE"
    else:
        overall_sentiment = "NEUTRAL"

    # Return the count of positive, negative, neutral chunks, and overall aggregated sentiment
    return JSONResponse(content={
        "positive_chunks": positive_chunks,
        "negative_chunks": negative_chunks,
        "neutral_chunks": neutral_chunks,
        "total_chunks": len(results),
        "overall_sentiment": overall_sentiment
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
