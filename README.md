# 券商研报 RAG 系统优化项目

围绕"基于研报的智能问答"做的三个相互衔接的实操：先用评估体系量化大模型回答质量，再设计 RAG 系统方案，最后用实验定位并优化检索瓶颈。

## 核心结论

| 模块 | 关键结果 |
|:---|:---|
| 回答质量评估 | 6 维度 1-5 分制，3 维自动评估 + 3 维人工评估。3 个模型综合均分 4.98 / 4.95 / 3.93，事实准确性和合规性是拉开差距的两个维度 |
| RAG 方案设计 | 结构感知切分 → Embedding → BM25 + 向量混合检索（RRF）→ Rerank → 带引文生成；识别出切分、多路融合、检索粒度三个核心挑战 |
| 检索优化实验 | Hybrid + Rerank 将 Recall@5 从 66.67% 提升到 76.67%；瓶颈已从"Query 表达不足"转向"跨行业相似章节混淆" |

## 目录结构

```
├── 01_llm_answer_evaluation/      大模型回答质量评估体系
│   ├── evaluation_framework.md    评估维度、1-5 分评分标准、自动/人工划分依据
│   ├── auto_evaluator.py          LLM-as-Judge 自动评估（两轮、按需核对）
│   ├── eval_results.csv           50 题 × 3 模型，自动评估三维结果
│   ├── eval_results_all.csv       六维度完整评分（含人工评分三维）
│   └── model_comparison.md        3 模型横向对比报告
├── 02_rag_solution_design/        RAG 问答系统方案
│   ├── rag_architecture.png       完整架构图（标注各模块输入输出）
│   └── rag_solution.md            技术选型与三个核心挑战
└── 03_retrieval_optimization/     检索优化实验
    ├── chunk.py                   研报按章节切分
    ├── embedding.py               Chunk 向量化
    ├── rag_retrieval.py           Baseline 与两种优化方案
    ├── experiment_results.csv     实验结果对比
    └── optimization_report.md     方法、结果、case 分析、下一步建议
```

## 01 大模型回答质量评估

**维度**：事实准确性、完整性、相关性（可自动评估）；表达质量、合规性、安全性（需人工评估）。每个维度的每个分值都有具体描述，且各维度定义互相独立，避免一个缺陷被重复扣分。

**自动评估**：采用 LLM-as-Judge。第一轮一次调用同时给三个维度打分，并要求列出回答中超出标准答案的"额外内容"；仅当存在额外内容时触发第二轮，用事实参考库核查是否构成编造，若编造则修正事实准确性分数。

**对比结果**（50 题 × 3 模型）：

| 模型 | 事实准确性 | 完整性 | 相关性 | 表达质量 | 合规性 | 安全性 | 综合 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Model_C | 5.00 | 4.90 | 5.00 | 5.00 | 5.00 | 5.00 | 4.98 |
| Model_A | 4.72 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 4.95 |
| Model_B | 2.58 | 5.00 | 4.02 | 4.00 | 2.96 | 5.00 | 3.93 |

三个模型的缺陷模式不同：C 事实全对，只在 ST 股票定义题上稳定漏掉"涨跌幅 5%"；A 存在固化的数字幻觉（科创板门槛 50 万答成 30 万、维持担保比例 130% 答成 120%），调 temperature 无法修复；B 每题都追加冗余内容，其中 26 条是编造的政策变化。详见 `model_comparison.md`，其中也列出了安全性无区分度、LLM 偶发误判等已知局限。

## 02 RAG 系统方案

离线构建知识库（PDF 解析 → 切分 → 向量化 → 入库），在线问答（查询向量化 → 混合检索 → 重排序 → 上下文组装 → 生成），每个环节的选型与理由见 `rag_solution.md`。针对金融研报场景总结了三个挑战：Chunking 策略影响召回质量；向量与 BM25 的分数不可直接相加，需 RRF 融合加 Rerank；检索粒度与生成所需上下文不一致，用 Parent-Child Chunking 解决。

## 03 检索优化实验

20 份研报按一级章节切为 100 个 Chunk，30 条标注问题，命中标准为同时命中目标研报和目标章节。

| 方法 | Recall@3 | Recall@5 |
|:---|---:|---:|
| Vector Retrieval（Baseline） | 60.00% | 66.67% |
| BM25 + Vector + Rerank | 66.67% | 76.67% |
| Query Rewrite + Vector | 63.33% | 70.00% |

case 分析显示，各行业研报的"风险提示""核心数据"章节文本高度模板化，检索能找到正确类型的章节，却区分不了是哪个行业的。Query Rewrite 对"行业 + 时间 + 指标"类问题有帮助，但对创新药等问题反而退化。下一步方向是 Metadata / Entity Filter → Hybrid Retrieval → Rerank，并按问题类型分层评估。

## 运行

```bash
pip install -r requirements.txt

# 01：评估（输入数据格式见 01_llm_answer_evaluation/data/README.md）
export DEEPSEEK_API_KEY=...
python 01_llm_answer_evaluation/auto_evaluator.py

# 03：检索实验（输入数据格式见 03_retrieval_optimization/data/README.md）
export DASHSCOPE_API_KEY=...
python 03_retrieval_optimization/chunk.py
python 03_retrieval_optimization/embedding.py
python 03_retrieval_optimization/rag_retrieval.py
```

Embedding 使用 `text-embedding-v2`，Query 改写使用 `qwen-plus`，Rerank 使用 `qwen3-rerank`（端点可通过环境变量 `RERANK_API` 覆盖）；评估 Judge 使用 `deepseek-chat`。

## 说明

输入数据（模拟问答、研报样本、测试问题）为合成数据，不随仓库提供，格式见各 `data/README.md`。仓库中的结果文件与报告为本人实验产出。
