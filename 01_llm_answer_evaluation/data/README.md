# 输入数据格式

数据为合成的模拟数据，不随仓库提供。运行 `auto_evaluator.py` 前，在本目录放置以下两个文件。

## model_outputs.json

50 道金融业务题，每题含标准答案与 3 个模型的回答：

```json
[
  {
    "id": "MO001",
    "question": "A股股票交易的印花税税率是多少？",
    "reference_answer": "卖出时按成交金额的0.1%征收印花税，买入不征收。",
    "model_a_answer": "...",
    "model_b_answer": "...",
    "model_c_answer": "...",
    "category": "交易规则"
  }
]
```

## financial_facts.json

金融事实参考库，用于第二轮"额外内容"编造核查：

```json
[
  {"fact_id": "F001", "category": "税费", "statement": "A股印花税卖出时按成交金额0.1%征收", "source": "财政部税务总局公告"}
]
```
