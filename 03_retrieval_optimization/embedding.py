import os
import json
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "chunks.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "embeddings.json")

API_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
API_KEY = os.environ["DASHSCOPE_API_KEY"]

BATCH_SIZE = 25


def get_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "text-embedding-v2",
        "input": {
            "texts": texts
        },
        "parameters": {
            "text_type": "document"
        }
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=data,
        timeout=60
    )

    response.raise_for_status()

    return response.json()["output"]["embeddings"]


with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

results = []

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]

    print(f"Embedding {i + 1}-{i + len(batch)} / {len(chunks)}")

    embeddings = get_embeddings([x["text"] for x in batch])

    for chunk, embedding in zip(batch, embeddings):
        results.append({
            "chunk_id": chunk["chunk_id"],
            "report_id": chunk["report_id"],
            "section": chunk["section"],
            "text": chunk["text"],
            "embedding": embedding["embedding"]
        })

    time.sleep(0.2)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)

print(f"完成：{len(results)} 个向量")
print(f"保存到：{OUTPUT_FILE}")