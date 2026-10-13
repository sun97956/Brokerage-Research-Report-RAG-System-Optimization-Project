# -*- coding: utf-8 -*-
"""
auto_evaluator.py
金融大模型回答质量评估 —— 自动化脚本

评估维度（3个可自动评估维度，均采用 LLM-as-Judge）：
  1. 事实准确性  2. 完整性  3. 相关性

策略（两轮、按需核对）：
  第一轮：一次调用同时给三个维度打分，并要求 LLM 指出模型回答中
          超出标准答案的"额外内容"
  第二轮：仅当存在额外内容时，才用金融事实参考库全量核对，判断额外内容
          是否构成编造；若构成编造，取两轮分数中的较低值修正事实准确性
  输出：eval_results.csv
"""

import os
import re
import csv
import json
import time
from openai import OpenAI

# ===================== 1. 路径配置 =====================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, "data")  # 输入数据格式见 data/README.md
QUESTIONS_FILE = os.path.join(DATA_DIR, "model_outputs.json")
FACTS_FILE = os.path.join(DATA_DIR, "financial_facts.json")
OUTPUT_CSV = os.path.join(BASE_PATH, "eval_results.csv")

# ===================== 2. DeepSeek 客户端 =====================
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)
MODEL_NAME = "deepseek-chat"

# ===================== 3. 读取数据 =====================
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions_data = json.load(f)

with open(FACTS_FILE, "r", encoding="utf-8") as f:
    facts_data = json.load(f)

# ===================== 4. 事实库整理 =====================
FACTS_ALL = "\n".join(f"- {fact['statement']}" for fact in facts_data)

# ===================== 5. 完整评分规则 =====================
SCORING_RULES = """
### 2.1 事实准确性
定义：仅衡量回答中数字、规则、定义的对错，不涉及多寡、完整度等其他方面。
5分：所有数字、规则、定义与标准答案/事实库完全一致，无任何编造信息
4分：核心事实正确，存在极微小瑕疵（如附加无害的无关说明），无编造假数据
3分：有一处明确的数字或规则错误，但大部分内容正确
2分：多处事实错误，数字和规则均有误，但回答内容仍与题目相关（含编造具体数据）
1分：满口胡言，凭空编造完全不存在的政策/数据，或所有事实全错

### 2.2 完整性
定义：仅衡量标准答案中的要点是否全部被覆盖，不涉及对错、通顺等其他方面。
5分：标准答案中的所有要点全部覆盖，无遗漏
4分：覆盖了主要要点，遗漏了次要细节
3分：覆盖了约一半的要点
2分：只覆盖了少量要点，大部分内容缺失
1分：严重遗漏，几乎没有覆盖问题要求的核心内容

### 2.3 相关性
定义：仅衡量回答是否围绕问题本身展开，不涉及内容对错、完整度等其他方面。
5分：完全紧扣问题，无无关信息，直击要点
4分：主要回答紧扣问题，有极少量无关补充但不影响理解
3分：基本相关，但部分内容偏离主题
2分：大部分内容偏离问题，仅少量相关
1分：完全答非所问，回答内容与问题毫无关系
"""

# ===================== 6. 第一轮：三维度打分 + 指出额外内容 =====================
def evaluate_main(question, reference, model_answer):
    prompt = f"""你是一个金融大模型回答质量评估专家。请严格依据以下评分标准对模型回答打分。

{SCORING_RULES}

【输入信息】
题目：{question}
标准答案：{reference}
模型回答：{model_answer}

【输出要求】
只输出以下四行，不要输出任何解释性废话：

事实准确性：X分
完整性：X分
相关性：X分
额外内容：<模型回答中标准答案里没有的内容，逐条列出原文；若没有则只写"无">"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用失败: {e}"


def parse_main(text):
    res = {"factuality": None, "completeness": None, "relevance": None, "extra": ""}

    m = re.search(r'事实准确性[:：]\s*(\d)', text)
    if m:
        res["factuality"] = int(m.group(1))

    m = re.search(r'完整性[:：]\s*(\d)', text)
    if m:
        res["completeness"] = int(m.group(1))

    m = re.search(r'相关性[:：]\s*(\d)', text)
    if m:
        res["relevance"] = int(m.group(1))

    m = re.search(r'额外内容[:：]\s*(.*)', text)
    if m:
        extra = m.group(1).strip()
        if extra and extra not in ("无", "无。", "没有", "无额外内容", "无额外内容。"):
            res["extra"] = extra

    return res

# ===================== 7. 第二轮：按需核对额外内容 =====================
def verify_extra(extra_text):
    """仅当存在额外内容时调用：用事实库全量核对是否构成编造"""
    prompt = f"""你是金融事实核查专家。

下面是一段模型回答中超出标准答案的"额外内容"。请判断它是否构成事实编造。

【金融事实参考库】
{FACTS_ALL}

【额外内容】
{extra_text}

【判断说明】
- 若额外内容包含具体的事实、数据、政策或规则（如"某年上调了X%"），且在参考库中找不到依据，属于编造。
- 若额外内容只是非事实性的提示语（如"建议咨询具体券商""以最新规定为准""相关规定可能调整"），
  不提出任何具体事实主张，不属于编造，即便参考库中没有对应记录。

【输出要求】
只输出以下两行：

判定：<只能写"有依据"或"无依据">
修正分数：<若判定为"无依据"，依据事实准确性评分标准给出修正后的1-5整数分；若判定为"有依据"，写"无需修正">"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用失败: {e}"


def parse_verify(text):
    fabricated = False
    new_score = None
    if re.search(r'判定[:：]\s*无依据', text):
        fabricated = True
    m = re.search(r'修正分数[:：]\s*(\d)', text)
    if m:
        new_score = int(m.group(1))
    return fabricated, new_score

# ===================== 8. 主流程：两层循环 =====================
def main():
    results = []

    total = len(questions_data)
    for idx, item in enumerate(questions_data, 1):
        qid = item["id"]
        question = item["question"]
        reference = item["reference_answer"]
        category = item["category"]   

        for field, model_name in [
            ("model_a_answer", "Model_A"),
            ("model_b_answer", "Model_B"),
            ("model_c_answer", "Model_C"),
        ]:
            answer = item[field]

            # 第一轮
            raw1 = evaluate_main(question, reference, answer)
            parsed = parse_main(raw1)

            fact = parsed["factuality"]
            comp = parsed["completeness"]
            rel = parsed["relevance"]
            extra = parsed["extra"]

            # 第二轮：仅在存在额外内容时核对
            if extra and fact is not None:
                raw2 = verify_extra(extra)
                fabricated, verify_score = parse_verify(raw2)
                if fabricated:
                    # 判定为编造：取两轮分数中的较低值（防止第二轮 LLM 给分过低）
                    if verify_score is not None:
                        fact = min(fact, verify_score)
                    else:
                        fact = min(fact, 2)   # 编造具体数据至少扣到2分

            results.append({
                "question_id": qid,
                "category": category,
                "model": model_name,
                "factuality_score": fact,
                "completeness_score": comp,
                "relevance_score": rel,
            })

            print(f"[{idx}/{total}] {qid} | {model_name} | 事实={fact} 完整={comp} 相关={rel}")
            time.sleep(0.3)

    # ===================== 9. 输出 CSV =====================
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "category", "model",
            "factuality_score", "completeness_score", "relevance_score"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)


    print(f"\n{'='*60}")
    print(f"完成！共 {len(results)} 条记录（{total} 题 × 3 模型）")
    print(f"CSV 已保存至: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()