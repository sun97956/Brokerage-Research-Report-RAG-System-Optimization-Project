# 券商研报 RAG 问答系统方案

## 一、全流程技术方案及选型

系统分为离线知识库构建与在线问答两部分

### 1. PDF 解析

使用结构化 PDF Parser 提取正文、标题、表格；扫描版通过 OCR 补充；并保留页码、章节等 Metadata。
**输出：**结构化文本/表格 + Metadata

### 2. 文本切分

采用结构感知 Chunking，优先按章节、子章节和段落切分；内容过长时再递归切分，并设置适当 Overlap

**输出：**Chunk + Metadata

### 3. 向量化

采用支持中文及中英混合语义的 Embedding Model（如 BGE-M3），通过 Recall@K 测试确定最终模型，然后将 Chunk 转换为向量

**输出：**Vector

### 4. 检索

采用 Hybrid Retrieval：Vector Search 负责语义召回，BM25 补充公司名称、股票代码、政策名称及精确数字等关键词匹配；两路结果通过 RRF 按排名融合，再由 Reranker 对候选 Chunk 进行精排。Vector Search 使用 HNSW 等索引提高检索效率

**输出：**Top-K 相关 Chunk

### 5. 生成

将用户问题、Top-K Chunk 和系统指令组装为 Prompt，交给 LLM 生成答案。要求模型仅基于检索内容回答，证据不足时拒答或说明不足，并返回报告引文

**输出：**答案 + 来源

## 二、金融研报场景的三个核心挑战

### 挑战 1：Chunking 策略会直接影响召回质量

Chunk 太小容易丢失完整观点，太大又会混入多个主题；固定长度、递归切分、语义切分各有取舍。研报本身章节结构清晰，因此优先采用结构感知切分，再根据过长程度递归拆分，并通过实际问题集比较不同 Chunk Size、Overlap 对 Recall 的影响

### 挑战 2：多路检索的结果如何融合和排序

Vector Search 擅长语义匹配，BM25 擅长精确关键词匹配，但两者 Score 不在同一尺度，不能直接相加；即使融合后，排名靠前的 Chunk 也不一定最相关

应对：采用 Vector Search + BM25 → RRF → Reranker，先扩大召回范围，再进行精排，在 Recall 与 Precision 之间取得平衡。

### 挑战 3：检索粒度与生成所需上下文不一致

小 Chunk 容易精准命中，但可能缺少完整背景；大 Chunk 信息完整，却会降低检索精度

应对：采用 Parent-Child Chunking：用 Child 建立向量索引并负责精准检索，命中后返回对应 Parent 或更大的上下文给 LLM，实现"找得准、看得全"，同时避免把大量无关内容塞入 Prompt