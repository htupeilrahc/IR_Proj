# 全功能微型文本搜索引擎 (Mini Full-Stack Search Engine)

纯 Python 实现的教学级搜索引擎，复刻经典 IR 理论（参考《Introduction to Information Retrieval》）

## 核心 IR 理论覆盖

- **SPIMI** 单次遍历内存索引 + 字母排序
- **位置倒排索引** 含 Gap Encoding / VByte 压缩理论注释
- **lnc.ltc** 严格 SMART 符号 TF-IDF 方案
- **双指针合并** O(x+y) 布尔 AND
- **DF 优化** 按 posting list 长度升序合并
- **短语查询** 基于位置偏移量检查
- **DAAT 累加打分** + 余弦相似度（归一化点积）
- **Min-Heap Top-K** 使用 `heapq` 提取，O(n log k)
- **Levenshtein 编辑距离** 拼写纠错

## 运行方式

本项目仅依赖 Python 标准库，无需安装任何第三方包。


运行后将依次执行 4 个测试演示：
1. **布尔 AND 查询** — 展示按 DF 优化的合并顺序
2. **短语查询** — 展示位置索引的精确匹配（含反例验证）
3. **拼写纠错** — 展示 "Did you mean...?" 提示
4. **排序检索** — 展示 lnc.ltc 打分，证明 IDF 惩罚常见词、提权罕见词

## 文件说明

| 文件 | 用途 |
|------|------|
| `mini_search_engine.py` | 完整可运行的搜索引擎源码（含全部模块 + 测试用例） |
| `promote.md` | 原始需求文档 |
| `README.md` | 本说明文件 |

## 运行步骤
- cd /Users/shaociliang/Downloads/university/信息检索/proj
- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

## 启动web服务
python3 app.py
打开浏览器访问： http://127.0.0.1:5001

## 1. 预处理与索引构建模块 (Text Processing & Indexing)
- **Tokenization**：实现基础的分词、转小写、去除基础标点。
- **SPIMI 思想**：模拟单次遍历内存索引思想，利用 Python `dict` 在内存中动态构建索引，并在构建完成后对 Term 键进行字母排序。
- **位置倒排索引 (Positional Inverted Index)**：数据结构必须不仅记录 docID，还要记录位置。格式类似于：`term -> {docID: [pos1, pos2], docID2: [pos1]}`。同时需维护 Document Frequency (DF)。
- **(理论扩展注释)**：请在倒排索引的代码类上方添加注释，简要说明在真实的工业级系统中，这里的 Postings List 应该如何使用 Gap Encoding (间距编码) 和 Variable Byte Code (可变字节编码) 进行压缩。

## 2. 向量打分与权重计算模块 (TF-IDF Weighting: lnc.ltc)
- 索引需支持为**向量空间模型 (VSM)** 提供基础数据，系统需实现严格的 TF-IDF 计分。
- 加权方案必须严格采用 SMART 符号表示法中的 **`lnc.ltc`** 方案：
  - **Document 权重 (`lnc`)**: 
    - `l` (Logarithmic TF): `1 + log10(tf)` (若 tf>0 否则 0)
    - `n` (No IDF): 文档端不使用 IDF 权重
    - `c` (Cosine normalization): 将文档向量除以其 L2 范数（需在建库时预计算或记录 norm）
  - **Query 权重 (`ltc`)**:
    - `l` (Logarithmic TF): `1 + log10(tf)`
    - `t` (IDF): `log10(N / df)`
    - `c` (Cosine normalization): 查询向量的余弦归一化

## 3. 查询与检索引擎 (Query Processing Engine)
系统需提供两种独立的查询模式：
- **模式 A: 精确布尔与短语检索 (Exact Boolean & Phrase Search)**
  - 实现基于“双指针法 (Two-pointer merge)”的线性时间 O(x+y) 倒排表合并算法，支持 `AND` 操作。
  - **查询优化**：多个 AND 条件时，必须按照 DF 从小到大升序合并。
  - **短语查询**：利用位置索引，实现基于位置偏移量的双指针检查，支持带双引号的精确短语匹配（如 `"stanford university"`）。
- **模式 B: 自由文本排序检索 (Free-text Ranked Retrieval)**
  - 计算查询向量与文档向量之间的余弦相似度。由于已做 `c` 归一化，请直接利用**点积 (Dot Product)** 快速打分。
  - 采用 **TAAT (Term-at-a-time)** 或 **DAAT** 的思想进行累加打分。
  - **Top-K 提取**: 必须使用最小堆 (Min-Heap, 例如 `heapq`) 提取得分最高的前 K 个文档，绝不能使用全量数组排序。

## 4. 容错检索模块 (Tolerant Retrieval)
- 实现一个简单的“非词拼写纠错” (Spelling Correction)。
- 当查询词不在词典中时，利用**编辑距离 (Levenshtein Distance)** 或 k-gram 找出最相似的 Top 1 候选词，作为 "Did you mean...?" 提示。
