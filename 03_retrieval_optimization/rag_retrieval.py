import os
import json
import time
import requests
import jieba
from rank_bm25 import BM25Okapi

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.json")
TEST_FILE = os.path.join(DATA_DIR, "test_questions.json")

BASELINE_OUTPUT = os.path.join(RESULT_DIR, "retrieval_results.json")
HYBRID_OUTPUT = os.path.join(RESULT_DIR, "hybrid_rerank_results.json")
REWRITE_OUTPUT = os.path.join(RESULT_DIR, "query_rewrite_results.json")
SUMMARY_CSV = os.path.join(RESULT_DIR, "experiment_results.csv")

API_KEY = os.environ["DASHSCOPE_API_KEY"]
EMBEDDING_API = (
    "https://dashscope.aliyuncs.com/"
    "api/v1/services/embeddings/text-embedding/text-embedding"
)
REWRITE_API = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
# Rerank 端点可通过环境变量 RERANK_API 覆盖
RERANK_API = os.environ.get(
    "RERANK_API",
    "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
)

EMBEDDING_MODEL = "text-embedding-v2"
REWRITE_MODEL = "qwen-plus"
RERANK_MODEL = "qwen3-rerank"
TOP_K = 10


# ============================================================
# 基础函数
# ============================================================
def post_json(url, data):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()


def get_query_embedding(text):
    data = {
        "model": EMBEDDING_MODEL,
        "input": {"texts": [text]},
        "parameters": {"text_type": "query"}
    }
    return post_json(EMBEDDING_API, data)["output"]["embeddings"][0]["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def is_hit(chunk, question):
    return (
        chunk["report_id"] == question["relevant_report"]
        and chunk["section"] == question["relevant_section"]
    )


def evaluate(results):
    total = len(results)
    hit3 = sum(x["hit_at_3"] for x in results)
    hit5 = sum(x["hit_at_5"] for x in results)
    return {
        "total_questions": total,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "recall_at_3": hit3 / total,
        "recall_at_5": hit5 / total
    }


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 1. Baseline：Vector Retrieval
# ============================================================
def vector_retrieval(chunks, questions, query_builder=lambda q: q):
    results = []

    for i, question in enumerate(questions, 1):
        qid = question["qid"]
        query = query_builder(question)
        print(f"[Vector {i}/{len(questions)}] {qid}")

        query_embedding = get_query_embedding(query)
        scored = []

        for chunk in chunks:
            score = cosine_similarity(query_embedding, chunk["embedding"])
            scored.append({
                "chunk_id": chunk["chunk_id"],
                "report_id": chunk["report_id"],
                "section": chunk["section"],
                "score": float(score)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top3 = scored[:3]
        top5 = scored[:5]

        results.append({
            "qid": qid,
            "question": question["question"],
            "query": query,
            "ground_truth": {
                "report_id": question["relevant_report"],
                "section": question["relevant_section"]
            },
            "top3": top3,
            "top5": top5,
            "hit_at_3": any(is_hit(x, question) for x in top3),
            "hit_at_5": any(is_hit(x, question) for x in top5)
        })

        time.sleep(0.2)

    return results


# ============================================================
# 2. Hybrid Retrieval：BM25 + Vector + Rerank
# ============================================================
def get_rerank(query, documents):
    data = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": len(documents)
    }
    return post_json(RERANK_API, data)["results"]


def hybrid_rerank_retrieval(chunks, questions):
    corpus = [list(jieba.cut(chunk["text"])) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    results = []

    for i, question in enumerate(questions, 1):
        qid = question["qid"]
        query = question["question"]
        print(f"[Hybrid {i}/{len(questions)}] {qid}")

        # BM25 Top10
        bm25_scores = bm25.get_scores(list(jieba.cut(query)))
        bm25_indices = sorted(
            range(len(chunks)),
            key=lambda x: bm25_scores[x],
            reverse=True
        )[:TOP_K]

        # Vector Top10
        query_embedding = get_query_embedding(query)
        vector_scores = [
            cosine_similarity(query_embedding, chunk["embedding"])
            for chunk in chunks
        ]
        vector_indices = sorted(
            range(len(chunks)),
            key=lambda x: vector_scores[x],
            reverse=True
        )[:TOP_K]

        # 合并候选集并去重
        candidate_indices = []
        seen = set()
        for idx in bm25_indices + vector_indices:
            chunk_id = chunks[idx]["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                candidate_indices.append(idx)

        candidates = [chunks[idx] for idx in candidate_indices]
        documents = [chunk["text"] for chunk in candidates]

        # Rerank
        rerank_results = get_rerank(query, documents)
        ranked = []
        for item in rerank_results:
            chunk = candidates[item["index"]]
            ranked.append({
                "chunk_id": chunk["chunk_id"],
                "report_id": chunk["report_id"],
                "section": chunk["section"],
                "score": float(item["relevance_score"])
            })

        top3 = ranked[:3]
        top5 = ranked[:5]

        results.append({
            "qid": qid,
            "question": query,
            "ground_truth": {
                "report_id": question["relevant_report"],
                "section": question["relevant_section"]
            },
            "candidate_count": len(candidates),
            "top3": top3,
            "top5": top5,
            "hit_at_3": any(is_hit(x, question) for x in top3),
            "hit_at_5": any(is_hit(x, question) for x in top5)
        })

        time.sleep(0.2)

    return results


# ============================================================
# 3. Strategy 2：Query Rewrite + Vector Retrieval
# ============================================================
def rewrite_query(question):
    prompt = f"""
你是一个金融研报检索查询改写器。

请将用户问题转换为适合金融研报检索的结构化查询。

要求：
1. 提取问题中的行业名称。
2. 判断用户主要对应的研报章节，只能从以下选项中选择：
   - 行业概况
   - 竞争格局
   - 核心数据
   - 风险提示
   - 投资建议
3. 提取问题中的时间信息。
4. 提取关键指标，例如营业收入、净利润、CR5、投资评级等。
5. 提取有助于检索的关键词。
6. 不得编造原问题中没有明确表达的具体事实、数字、公司名称或风险因素。
7. 只输出 JSON，不要输出解释文字。

输出格式：
{{
  "industry": "",
  "section": "",
  "time": "",
  "metrics": [],
  "keywords": []
}}

用户问题：
{question}
"""

    data = {
        "model": REWRITE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    content = post_json(REWRITE_API, data)["choices"][0]["message"]["content"].strip()

    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    return json.loads(content)


def build_rewritten_query(rewrite):
    parts = []

    for key in ["industry", "section", "time"]:
        if rewrite.get(key):
            parts.append(rewrite[key])

    parts.extend(rewrite.get("metrics", []))
    parts.extend(rewrite.get("keywords", []))

    # 去除重复词，避免改写结果重复强化同一关键词
    return " ".join(dict.fromkeys(parts))


def query_rewrite_retrieval(chunks, questions):
    results = []

    for i, question in enumerate(questions, 1):
        qid = question["qid"]
        original_query = question["question"]
        print(f"[Rewrite {i}/{len(questions)}] {qid}")

        rewrite = rewrite_query(original_query)
        rewritten_query = build_rewritten_query(rewrite)

        query_embedding = get_query_embedding(rewritten_query)
        scored = []

        for chunk in chunks:
            score = cosine_similarity(query_embedding, chunk["embedding"])
            scored.append({
                "chunk_id": chunk["chunk_id"],
                "report_id": chunk["report_id"],
                "section": chunk["section"],
                "score": float(score)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top3 = scored[:3]
        top5 = scored[:5]

        results.append({
            "qid": qid,
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "rewrite": rewrite,
            "ground_truth": {
                "report_id": question["relevant_report"],
                "section": question["relevant_section"]
            },
            "top3": top3,
            "top5": top5,
            "hit_at_3": any(is_hit(x, question) for x in top3),
            "hit_at_5": any(is_hit(x, question) for x in top5)
        })

        time.sleep(0.2)

    return results


# ============================================================
# 4. Main：运行三种检索方案
# ============================================================
def main():
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"加载 Chunk：{len(chunks)}")
    print(f"测试问题：{len(questions)}")

    # Baseline
    baseline_results = vector_retrieval(chunks, questions)
    baseline_metrics = evaluate(baseline_results)
    save_json(
        BASELINE_OUTPUT,
        {**baseline_metrics, "results": baseline_results}
    )

    # Hybrid + Rerank
    hybrid_results = hybrid_rerank_retrieval(chunks, questions)
    hybrid_metrics = evaluate(hybrid_results)
    save_json(
        HYBRID_OUTPUT,
        {**hybrid_metrics, "results": hybrid_results}
    )

    # Query Rewrite + Vector
    rewrite_results = query_rewrite_retrieval(chunks, questions)
    rewrite_metrics = evaluate(rewrite_results)
    save_json(
        REWRITE_OUTPUT,
        {**rewrite_metrics, "results": rewrite_results}
    )

    # 汇总
    print("\n" + "=" * 60)
    print("Retrieval Experiment Results")
    print("=" * 60)
    print(
        f"Baseline          Recall@3={baseline_metrics['recall_at_3']:.2%}  "
        f"Recall@5={baseline_metrics['recall_at_5']:.2%}"
    )
    print(
        f"Hybrid + Rerank   Recall@3={hybrid_metrics['recall_at_3']:.2%}  "
        f"Recall@5={hybrid_metrics['recall_at_5']:.2%}"
    )
    print(
        f"Query Rewrite     Recall@3={rewrite_metrics['recall_at_3']:.2%}  "
        f"Recall@5={rewrite_metrics['recall_at_5']:.2%}"
    )
    print("=" * 60)

    import csv
    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Description", "Recall@3", "Recall@5"])
        for name, desc, m in [
            ("Baseline", "Vector Retrieval", baseline_metrics),
            ("Strategy 1", "BM25 + Vector + Rerank", hybrid_metrics),
            ("Strategy 2", "Query Rewrite + Vector Retrieval", rewrite_metrics),
        ]:
            w.writerow([name, desc, f"{m['recall_at_3']:.2%}", f"{m['recall_at_5']:.2%}"])


if __name__ == "__main__":
    main()
