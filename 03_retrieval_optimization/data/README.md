# 输入数据格式

数据为合成的模拟数据，不随仓库提供。运行顺序：`chunk.py` → `embedding.py` → `rag_retrieval.py`。

## reports/*.txt

20 份行业研报，每份一个 TXT。开头为元信息，正文按一级章节组织：

```
报告编号：REP01
报告标题：新能源汽车行业2024年度投资策略报告
分析师：...
发布日期：2024-05-04
投资评级：买入

一、行业概况
...
二、竞争格局
...
三、核心数据
...
四、风险提示
...
五、投资建议
...
免责声明：...
```

`chunk.py` 按一级章节切分，20 份研报共得到 100 个 Chunk，输出 `chunks.json`。

## test_questions.json

30 条标注好的"问题-相关段落"测试集：

```json
[
  {
    "qid": "RT01",
    "question": "新能源汽车行业的投资建议和关注主线是什么？",
    "relevant_report": "REP01",
    "relevant_section": "投资建议",
    "industry": "新能源汽车"
  }
]
```

命中标准：召回的 Chunk 同时满足 `report_id == relevant_report` 且 `section == relevant_section`。

## 其他

`embedding.py` 生成 `embeddings.json`（含向量，体积大，已在 `.gitignore` 中）。运行结果写入 `../results/`。
