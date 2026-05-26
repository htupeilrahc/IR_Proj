# 全功能微型文本搜索引擎 (Mini Full-Stack Search Engine)

纯 Python 实现的教学级搜索引擎，完整覆盖 IR 课程全部章节（参考《Introduction to Information Retrieval》）。

---

## 核心 IR 理论覆盖

### Part 1 — 搜索引擎基础
- **SPIMI** 单次遍历内存索引 + 字母排序
- **位置倒排索引** 含 Gap Encoding / VByte 压缩理论注释
- **lnc.ltc** 严格 SMART 符号 TF-IDF 方案
- **双指针合并** O(x+y) 布尔 AND
- **DF 优化** 按 posting list 长度升序合并
- **短语查询** 基于位置偏移量检查
- **DAAT 累加打分** + 余弦相似度（归一化点积）
- **Min-Heap Top-K** 使用 `heapq` 提取，O(n log k)
- **Levenshtein 编辑距离** 拼写纠错

### Part 2 — 高级检索与机器学习 (NEW)
- **检索评价** Precision/Recall/F1, MAP, NDCG, 11点插值 P-R 曲线 (ch8)
- **BM25 概率检索** TF 饱和 + 文档长度归一化 (ch9&11)
- **语言模型检索** Dirichlet / Jelinek-Mercer 平滑 (ch9&11)
- **文本分类** Naïve Bayes, k-NN, Rocchio 质心分类 (ch13&14)
- **学习排序 LTR** Pointwise + 决策树回归 + 6维特征工程 (ch15)
- **链接分析** PageRank (Random Surfer) / HITS (Hubs & Authorities) (ch21)
- **个性化检索** 用户画像构建 + α 加权融合重排序

---

## 运行方式

### 1. 命令行测试

```bash
# Part 1 测试 (布尔/短语/TF-IDF/拼写纠错)
python3 mini_search_engine.py

# Part 2 测试 (BM25/LM/评价/分类/LTR/PageRank/个性化)
python3 mini_search_engine_v2.py
```

### 2. Web 可视化界面

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Web 服务
python3 app.py

# 浏览器访问
# http://127.0.0.1:5001
```

Web 界面提供 **11 个功能 Tab**，覆盖前后两个学期的全部内容。

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `mini_search_engine.py` | Part 1 核心模块 + 测试用例 |
| `mini_search_engine_v2.py` | Part 2 核心模块 + 测试用例 |
| `app.py` | Flask Web 后端 (全部 API) |
| `templates/index_v2.html` | 前端页面 (11个功能 Tab) |
| `requirements.txt` | Python 依赖 (仅 Flask) |
| `promote.md` | 原始需求文档 |

---

## 环境要求

- Python 3.7+ (标准库即可运行核心模块)
- Flask >= 3.0 (仅 Web 界面需要)

---

## Part 1 模块详解

### 1. 预处理与索引构建模块 (Text Processing & Indexing)
- **Tokenization**：实现基础的分词、转小写、去除基础标点。
- **SPIMI 思想**：模拟单次遍历内存索引思想，利用 Python `dict` 在内存中动态构建索引，并在构建完成后对 Term 键进行字母排序。
- **位置倒排索引 (Positional Inverted Index)**：数据结构不仅记录 docID，还记录位置。格式：`term -> {docID: [pos1, pos2], docID2: [pos1]}`。同时维护 Document Frequency (DF)。
- **(理论扩展注释)**：代码注释中说明了在真实工业级系统中，Postings List 应如何使用 Gap Encoding (间距编码) 和 Variable Byte Code (可变字节编码) 进行压缩。

### 2. 向量打分与权重计算模块 (TF-IDF Weighting: lnc.ltc)
- 严格采用 SMART 符号表示法中的 **`lnc.ltc`** 方案：
  - **Document 权重 (`lnc`)**: `l` (Logarithmic TF: `1 + log10(tf)`), `n` (No IDF), `c` (Cosine normalization)
  - **Query 权重 (`ltc`)**: `l` (Logarithmic TF), `t` (IDF: `log10(N / df)`), `c` (Cosine normalization)

### 3. 查询与检索引擎 (Query Processing Engine)
- **模式 A: 精确布尔与短语检索**
  - 双指针法 O(x+y) 倒排表合并，支持 `AND` 操作。
  - **DF 优化**：多个 AND 条件按 DF 升序合并。
  - **短语查询**：利用位置索引实现基于位置偏移量的精确短语匹配（如 `"stanford university"`）。
- **模式 B: 自由文本排序检索**
  - 利用已归一化向量的**点积 (Dot Product)** 快速计算余弦相似度。
  - **DAAT** 累加打分 + **Min-Heap** 提取 Top-K。

### 4. 容错检索模块 (Tolerant Retrieval)
- 当查询词不在词典中时，利用 **Levenshtein 编辑距离** 找出最相似的候选词，作为 "Did you mean...?" 提示。

---

## Part 2 模块速查

| 类名 | 章节 | 功能 |
|------|------|------|
| `EvaluationMetrics` | ch8 | P/R/F1, P@K, MAP, MRR, DCG, NDCG, 11点P-R曲线 |
| `BM25Scorer` | ch9&11 | Okapi BM25 (k₁, b, avgdl) |
| `LanguageModelScorer` | ch9&11 | 查询似然 + Dirichlet / Jelinek-Mercer 平滑 |
| `NaiveBayesClassifier` | ch13 | 多项式朴素贝叶斯 + 拉普拉斯平滑 |
| `KNNClassifier` | ch14 | TF-IDF 向量 + 余弦相似度 + 多数投票 |
| `RocchioClassifier` | ch14 | 质心分类器 |
| `DecisionTreeRanker` | ch15 | 回归决策树 (MSE 分裂) |
| `LearningToRank` | ch15 | Pointwise LTR + 6维特征工程 |
| `PageRank` | ch21 | Random Surfer 迭代 (d=0.85) |
| `HITS` | ch21 | Hub / Authority 相互增强 |
| `SearchPersonalizer` | — | 用户画像 + α 加权融合重排序 |
