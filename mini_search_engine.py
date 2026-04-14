"""
全功能微型文本搜索引擎 (Mini Full-Stack Search Engine)
======================================================
纯 Python 实现，复刻经典 IR 理论（参考《Introduction to Information Retrieval》），涵盖：
- SPIMI 思想的位置倒排索引构建
- lnc.ltc TF-IDF 加权方案 (SMART 符号)
- 布尔 AND 查询（双指针合并，DF 优化）
- 短语查询（位置索引 + 位置偏移量检查）
- 自由文本排序检索（余弦相似度 + Top-K 最小堆）
- 拼写纠错（Levenshtein 编辑距离）

所有核心逻辑均使用 Python 原生数据结构手工实现，不依赖任何第三方库。
"""

import math
import heapq
import string
from collections import defaultdict


# ============================================================
# 模块1: 文本预处理器 (Text Preprocessor)
# ============================================================

class TextPreprocessor:
    """
    负责文本的分词、转小写、去标点等预处理操作。
    对应 IR 理论中的 Tokenization 步骤。
    """

    def __init__(self):
        # 构建标点符号转换表，用于移除所有基础标点
        self._punct_table = str.maketrans('', '', string.punctuation)

    def tokenize(self, text: str) -> list:
        """
        对输入文本进行分词处理。

        步骤：
        1. 转小写 (Case folding)
        2. 去除标点 (Punctuation removal)
        3. 按空白符切分 (Whitespace tokenization)

        返回: [(term, position), ...] 带位置信息的 token 列表
        """
        # 转小写
        text = text.lower()
        # 去除标点
        text = text.translate(self._punct_table)
        # 按空白符分词，同时记录位置
        tokens = []
        position = 0
        for token in text.split():
            if token:  # 过滤空字符串
                tokens.append((token, position))
                position += 1
        return tokens


# ============================================================
# 模块2: 位置倒排索引 + SPIMI 索引构建器
# (Positional Inverted Index & SPIMI Indexer)
# ============================================================

"""
【理论扩展注释 - 倒排表压缩 (Postings List Compression)】

在真实的工业级搜索引擎（如 Lucene / Elasticsearch）中，
Postings List 会使用以下压缩技术以大幅降低磁盘和内存占用：

1. Gap Encoding（间距编码）：
   - 原始 Postings List 存储的是有序 docID 列表，如 [10, 23, 37, 85]
   - Gap Encoding 只存储相邻 docID 的差值 (gap): [10, 13, 14, 48]
   - 由于 gap 通常较小，可以用更少的位来表示

2. Variable Byte Code（可变字节编码，VByte）：
   - 对每个 gap 值用变长字节表示
   - 每个字节的最高位 (continuation bit) 标记是否还有后续字节：
     * 最高位=1 表示这是最后一个字节
     * 最高位=0 表示后续还有字节
   - 例如: gap=130 -> 需要 2 个字节: 00000001 10000010
   - 小 gap 只需 1 字节，大 gap 才需多字节，平均压缩率极高

3. 其他高级编码方案：
   - Gamma Encoding: Elias gamma 编码，适合 gap 分布偏斜的情况
   - PForDelta: 批量压缩，SIMD 友好，现代搜索引擎常用
   - Simple9/Simple16: 将多个小整数打包进一个 32 位字中

本教学系统为简化起见，直接使用 Python dict 存储，不做压缩。
"""


class PositionalInvertedIndex:
    """
    位置倒排索引数据结构。

    核心结构: term -> {docID: [pos1, pos2, ...], docID2: [pos1, ...]}
    同时维护:
    - df (Document Frequency): 每个 term 出现在多少个文档中
    - doc_count (N): 文档总数
    - doc_norms: 每个文档向量的 L2 范数（用于 lnc 的 c 归一化）
    """

    def __init__(self):
        # 核心倒排索引: term -> {docID: [positions]}
        self.index = {}
        # 文档频率: term -> df
        self.df = {}
        # 文档总数
        self.doc_count = 0
        # 文档标题/原文映射
        self.documents = {}
        # 文档向量的 L2 范数（lnc 中 c 归一化所需）
        self.doc_norms = {}

    def get_postings(self, term: str) -> dict:
        """获取某个 term 的完整 postings list (含位置信息)"""
        return self.index.get(term, {})

    def get_doc_ids(self, term: str) -> list:
        """获取某个 term 的有序 docID 列表（用于布尔查询的双指针合并）"""
        return sorted(self.index.get(term, {}).keys())

    def get_df(self, term: str) -> int:
        """获取 term 的文档频率 (Document Frequency)"""
        return self.df.get(term, 0)

    def get_tf(self, term: str, doc_id: int) -> int:
        """获取 term 在指定文档中的词频 (Term Frequency)"""
        postings = self.index.get(term, {})
        if doc_id in postings:
            return len(postings[doc_id])  # 位置列表长度即为 TF
        return 0

    def vocabulary(self) -> list:
        """返回按字母序排列的词典（体现 SPIMI 最终排序步骤）"""
        return sorted(self.index.keys())


class SPIMIIndexer:
    """
    基于 SPIMI (Single-Pass In-Memory Indexing) 思想的索引构建器。

    SPIMI 核心思想：
    - 单次遍历文档集合
    - 在内存中使用 dict 动态构建倒排索引（无需预先知道词典大小）
    - 遇到新 term 时直接创建新的 postings list
    - 遍历完成后对 term 键按字母序排序

    在工业系统中，当内存满时会将当前块写入磁盘 (block)，
    最后再做多路归并 (merge)。本教学版本在内存中一次性完成。
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def build_index(self, documents: dict) -> PositionalInvertedIndex:
        """
        构建位置倒排索引。

        参数:
            documents: {doc_id: text_content, ...}

        返回:
            PositionalInvertedIndex 实例
        """
        idx = PositionalInvertedIndex()
        idx.doc_count = len(documents)
        idx.documents = documents

        # ---- SPIMI 核心: 单次遍历，动态构建 ----
        for doc_id, text in documents.items():
            tokens = self.preprocessor.tokenize(text)

            for term, position in tokens:
                # 遇到新 term -> 创建新 postings list（SPIMI 关键步骤）
                if term not in idx.index:
                    idx.index[term] = {}

                # 遇到新 docID -> 创建新位置列表
                if doc_id not in idx.index[term]:
                    idx.index[term][doc_id] = []

                # 记录 term 在该文档中的位置
                idx.index[term][doc_id].append(position)

        # ---- SPIMI 收尾: 对 term 键按字母序排序 ----
        # Python dict 在 3.7+ 保持插入顺序，重建为有序 dict
        idx.index = dict(sorted(idx.index.items()))

        # ---- 计算文档频率 (DF) ----
        for term, postings in idx.index.items():
            idx.df[term] = len(postings)

        # ---- 预计算文档向量 L2 范数 (lnc 中的 c 归一化) ----
        # 此处体现了 lnc.ltc 中文档端的权重计算:
        #   l: logarithmic TF = 1 + log10(tf) if tf > 0 else 0
        #   n: no IDF (文档端不用 IDF)
        #   c: cosine normalization (除以 L2 范数)
        self._precompute_doc_norms(idx)

        return idx

    def _precompute_doc_norms(self, idx: PositionalInvertedIndex):
        """
        预计算每个文档的 L2 范数，用于 lnc 方案中的 c (cosine) 归一化。

        文档权重 = 1 + log10(tf)  (lnc 中的 l)
        L2 范数 = sqrt(Σ (1+log10(tf))^2)
        """
        doc_norm_sq = defaultdict(float)

        for term, postings in idx.index.items():
            for doc_id, positions in postings.items():
                tf = len(positions)
                # lnc 的 l: logarithmic TF
                if tf > 0:
                    w = 1.0 + math.log10(tf)
                else:
                    w = 0.0
                doc_norm_sq[doc_id] += w * w

        # 取平方根得到 L2 范数
        for doc_id in doc_norm_sq:
            idx.doc_norms[doc_id] = math.sqrt(doc_norm_sq[doc_id])


# ============================================================
# 模块3: TF-IDF 打分器 (lnc.ltc 方案)
# ============================================================

class TFIDFScorer:
    """
    实现严格的 SMART lnc.ltc TF-IDF 加权方案。

    SMART 符号表示: ddd.qqq (文档端.查询端)
    - lnc (文档): l=logarithmic TF, n=no IDF, c=cosine norm
    - ltc (查询): l=logarithmic TF, t=IDF, c=cosine norm
    """

    @staticmethod
    def log_tf(tf: int) -> float:
        """
        Logarithmic TF (l): 对应 lnc 和 ltc 中的 'l'
        公式: 1 + log10(tf) if tf > 0, else 0
        """
        if tf > 0:
            return 1.0 + math.log10(tf)
        return 0.0

    @staticmethod
    def idf(N: int, df: int) -> float:
        """
        Inverse Document Frequency (t): 对应 ltc 中的 't'
        公式: log10(N / df)
        """
        if df == 0:
            return 0.0
        return math.log10(N / df)

    @staticmethod
    def compute_doc_weight(tf: int) -> float:
        """
        计算文档端权重 (lnc):
        - l: 1 + log10(tf)
        - n: 不乘 IDF
        - c: 归一化在检索时通过除以预计算的 doc_norm 实现

        此处返回未归一化的权重，归一化在打分时完成。
        """
        return TFIDFScorer.log_tf(tf)

    @staticmethod
    def compute_query_weight(tf: int, N: int, df: int) -> float:
        """
        计算查询端权重 (ltc):
        - l: 1 + log10(tf)
        - t: log10(N / df)
        - c: 归一化在查询处理时对整个查询向量完成

        此处返回 l*t (未归一化)，归一化在查询处理时完成。
        """
        return TFIDFScorer.log_tf(tf) * TFIDFScorer.idf(N, df)


# ============================================================
# 模块4: 布尔查询引擎 (Boolean & Phrase Query Engine)
# ============================================================

class BooleanQueryEngine:
    """
    精确布尔与短语检索引擎。

    支持:
    - AND 布尔查询（双指针合并，DF 优化排序）
    - 短语查询（基于位置索引的位置偏移量检查）
    """

    def __init__(self, index: PositionalInvertedIndex):
        self.index = index
        self.preprocessor = TextPreprocessor()

    @staticmethod
    def _intersect_two_postings(list1: list, list2: list) -> list:
        """
        双指针法 (Two-pointer merge) 实现两个有序 docID 列表的交集。

        时间复杂度: O(x + y)，其中 x, y 分别为两个列表的长度。
        这是 IR 中布尔 AND 操作的经典实现。
        """
        result = []
        i, j = 0, 0
        while i < len(list1) and j < len(list2):
            if list1[i] == list2[j]:
                result.append(list1[i])
                i += 1
                j += 1
            elif list1[i] < list2[j]:
                i += 1
            else:
                j += 1
        return result

    def boolean_and_query(self, query: str) -> list:
        """
        执行布尔 AND 查询。

        【查询优化】多个 AND 条件时，按 DF 从小到大排序后依次合并，
        确保中间结果尽可能小，减少比较次数。
        这是教材中经典的"按 posting list 长度升序处理"优化策略。
        """
        tokens = self.preprocessor.tokenize(query)
        terms = [t[0] for t in tokens]

        if not terms:
            return []

        # 按 DF 升序排序（查询优化: DF 小的 term 先处理）
        terms_sorted = sorted(terms, key=lambda t: self.index.get_df(t))

        # 从 DF 最小的 term 开始
        result = self.index.get_doc_ids(terms_sorted[0])

        # 依次与后续 term 的 postings 做 AND 合并
        for term in terms_sorted[1:]:
            if not result:  # 提前终止: 中间结果为空则无需继续
                break
            next_list = self.index.get_doc_ids(term)
            result = self._intersect_two_postings(result, next_list)

        return result

    def phrase_query(self, phrase: str) -> list:
        """
        短语查询：利用位置索引实现精确短语匹配。

        算法思路：
        1. 先用布尔 AND 找到所有包含全部查询词的文档
        2. 在候选文档中，检查各词的位置是否构成连续序列
           (即 term[i] 的位置 = term[0] 的位置 + i)

        使用双指针法检查位置偏移，确保高效。
        """
        tokens = self.preprocessor.tokenize(phrase)
        terms = [t[0] for t in tokens]

        if not terms:
            return []

        if len(terms) == 1:
            return self.index.get_doc_ids(terms[0])

        # 第一步: 布尔 AND 找到候选文档
        candidate_docs = self.index.get_doc_ids(terms[0])
        for term in terms[1:]:
            candidate_docs = self._intersect_two_postings(
                candidate_docs, self.index.get_doc_ids(term)
            )

        # 第二步: 在候选文档中做位置检查
        result = []
        for doc_id in candidate_docs:
            if self._check_phrase_positions(terms, doc_id):
                result.append(doc_id)

        return result

    def _check_phrase_positions(self, terms: list, doc_id: int) -> bool:
        """
        基于位置偏移量的双指针检查。

        对于短语 "w1 w2 w3"，需要找到文档中某个位置 p 使得:
        - w1 出现在位置 p
        - w2 出现在位置 p+1
        - w3 出现在位置 p+2
        """
        # 获取第一个词的位置列表
        positions_first = self.index.get_postings(terms[0]).get(doc_id, [])

        for start_pos in positions_first:
            # 检查后续每个词是否出现在预期位置
            match = True
            for offset, term in enumerate(terms[1:], 1):
                positions = self.index.get_postings(term).get(doc_id, [])
                target_pos = start_pos + offset
                # 二分查找目标位置（位置列表已排序）
                if not self._binary_search(positions, target_pos):
                    match = False
                    break
            if match:
                return True
        return False

    @staticmethod
    def _binary_search(sorted_list: list, target: int) -> bool:
        """在有序列表中二分查找目标值"""
        lo, hi = 0, len(sorted_list) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_list[mid] == target:
                return True
            elif sorted_list[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return False


# ============================================================
# 模块5: 排序检索引擎 (Ranked Retrieval Engine)
# ============================================================

class RankedRetrievalEngine:
    """
    自由文本排序检索引擎。

    使用 lnc.ltc 方案计算查询-文档的余弦相似度。
    由于文档和查询向量均已做 cosine 归一化 (c)，
    相似度计算简化为归一化向量的点积 (Dot Product)。

    采用 DAAT (Document-at-a-time) 思想进行累加打分，
    使用最小堆 (Min-Heap) 提取 Top-K 结果。
    """

    def __init__(self, index: PositionalInvertedIndex):
        self.index = index
        self.scorer = TFIDFScorer()
        self.preprocessor = TextPreprocessor()

    def search(self, query: str, k: int = 10) -> list:
        """
        执行自由文本排序检索，返回 Top-K 结果。

        算法流程:
        1. 解析查询，计算查询向量权重 (ltc)
        2. 遍历查询中每个 term，对包含该 term 的文档累加分数 (DAAT)
        3. 使用最小堆维护 Top-K

        返回: [(score, doc_id), ...] 按分数降序排列
        """
        tokens = self.preprocessor.tokenize(query)
        if not tokens:
            return []

        # ---- 步骤1: 计算查询向量 (ltc) ----
        # 统计查询中各 term 的 TF
        query_tf = defaultdict(int)
        for term, _ in tokens:
            query_tf[term] += 1

        N = self.index.doc_count

        # 计算查询权重 (l * t，未归一化)
        query_weights = {}
        for term, tf in query_tf.items():
            df = self.index.get_df(term)
            if df > 0:  # 只处理词典中存在的 term
                # ltc: l (log TF) * t (IDF)
                w = self.scorer.compute_query_weight(tf, N, df)
                query_weights[term] = w

        if not query_weights:
            return []

        # 此处体现了 ltc 中的 c 归一化: 查询向量除以其 L2 范数
        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
        if query_norm > 0:
            for term in query_weights:
                query_weights[term] /= query_norm

        # ---- 步骤2: DAAT 累加打分 ----
        # 遍历查询中每个 term，对匹配文档累加 doc_weight * query_weight
        scores = defaultdict(float)

        for term, q_weight in query_weights.items():
            postings = self.index.get_postings(term)
            for doc_id, positions in postings.items():
                tf = len(positions)
                # lnc 的 l: logarithmic TF
                doc_weight = self.scorer.compute_doc_weight(tf)

                # lnc 的 c: 除以预计算的文档 L2 范数
                doc_norm = self.index.doc_norms.get(doc_id, 1.0)
                if doc_norm > 0:
                    doc_weight /= doc_norm

                # 累加点积分量 (已归一化的向量点积即为余弦相似度)
                scores[doc_id] += doc_weight * q_weight

        # ---- 步骤3: Top-K 提取 (最小堆) ----
        # 使用 heapq 维护大小为 K 的最小堆
        # 堆中存储 (score, doc_id)，堆顶是最小分数
        # 这确保了 O(n log k) 的时间复杂度，而非 O(n log n) 的全排序
        min_heap = []
        for doc_id, score in scores.items():
            if len(min_heap) < k:
                heapq.heappush(min_heap, (score, doc_id))
            elif score > min_heap[0][0]:
                heapq.heapreplace(min_heap, (score, doc_id))

        # 从堆中取出结果并按分数降序排列
        results = sorted(min_heap, key=lambda x: -x[0])
        return results


# ============================================================
# 模块6: 拼写纠错模块 (Spelling Correction)
# ============================================================

class SpellingCorrector:
    """
    基于编辑距离 (Levenshtein Distance) 的非词拼写纠错。

    当查询词不在词典中时，计算该词与词典中所有词的编辑距离，
    找出最相似的 Top-1 候选词，作为 "Did you mean...?" 提示。

    编辑距离定义: 将一个字符串转换为另一个字符串所需的
    最少单字符编辑操作数（插入、删除、替换）。
    """

    def __init__(self, vocabulary: list):
        self.vocabulary = vocabulary

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        计算两个字符串之间的 Levenshtein 编辑距离。

        使用动态规划 (DP):
        dp[i][j] = min(
            dp[i-1][j] + 1,      # 删除
            dp[i][j-1] + 1,      # 插入
            dp[i-1][j-1] + cost  # 替换 (cost=0 若字符相同, 否则 1)
        )
        """
        m, n = len(s1), len(s2)
        # 创建 DP 矩阵
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # 初始化边界
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # 填充 DP 矩阵
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,       # 删除
                    dp[i][j - 1] + 1,        # 插入
                    dp[i - 1][j - 1] + cost  # 替换
                )

        return dp[m][n]

    def correct(self, term: str) -> str:
        """
        为不在词典中的查询词找到最相似的候选词。

        返回编辑距离最小的 Top-1 词。
        若有多个候选编辑距离相同，选择字母序最小的。
        """
        if term in self.vocabulary:
            return term  # 词在词典中，无需纠错

        best_candidate = None
        best_distance = float('inf')

        for vocab_term in self.vocabulary:
            dist = self.levenshtein_distance(term, vocab_term)
            if dist < best_distance or (dist == best_distance and
                    (best_candidate is None or vocab_term < best_candidate)):
                best_distance = dist
                best_candidate = vocab_term

        return best_candidate


# ============================================================
# 模块7: 主搜索引擎 (Mini Search Engine - Facade)
# ============================================================

class MiniSearchEngine:
    """
    全功能微型搜索引擎的门面类 (Facade)。

    整合所有模块，提供统一的搜索接口:
    - 索引构建
    - 布尔 AND 查询
    - 短语查询
    - 排序检索
    - 拼写纠错
    """

    def __init__(self):
        self.indexer = SPIMIIndexer()
        self.index = None
        self.boolean_engine = None
        self.ranked_engine = None
        self.spell_corrector = None

    def build(self, documents: dict):
        """构建索引并初始化所有查询引擎"""
        print("=" * 60)
        print("  构建索引中 (Building Index)...")
        print("=" * 60)

        self.index = self.indexer.build_index(documents)
        self.boolean_engine = BooleanQueryEngine(self.index)
        self.ranked_engine = RankedRetrievalEngine(self.index)
        self.spell_corrector = SpellingCorrector(self.index.vocabulary())

        print(f"  文档数量 (N): {self.index.doc_count}")
        print(f"  词典大小 (|V|): {len(self.index.vocabulary())}")
        print(f"  索引构建完成！词典已按字母序排列 (SPIMI 排序步骤)")
        print()

    def boolean_search(self, query: str) -> list:
        """布尔 AND 查询"""
        return self.boolean_engine.boolean_and_query(query)

    def phrase_search(self, phrase: str) -> list:
        """短语查询"""
        return self.boolean_engine.phrase_query(phrase)

    def ranked_search(self, query: str, k: int = 5) -> list:
        """排序检索 (Top-K)"""
        return self.ranked_engine.search(query, k)

    def spell_check(self, term: str) -> str:
        """拼写纠错"""
        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize(term)
        if tokens:
            term_lower = tokens[0][0]
        else:
            term_lower = term.lower()
        return self.spell_corrector.correct(term_lower)

    def print_index_sample(self, max_terms: int = 10):
        """打印索引样本，展示位置倒排索引结构"""
        print("=" * 60)
        print("  位置倒排索引样本 (Positional Inverted Index Sample)")
        print("=" * 60)
        vocab = self.index.vocabulary()
        for term in vocab[:max_terms]:
            postings = self.index.get_postings(term)
            df = self.index.get_df(term)
            print(f"  '{term}' [DF={df}]:")
            for doc_id, positions in sorted(postings.items()):
                print(f"    Doc {doc_id}: positions = {positions}")
        if len(vocab) > max_terms:
            print(f"  ... (共 {len(vocab)} 个 terms，仅展示前 {max_terms} 个)")
        print()


# ============================================================
# 测试用例 (Test Cases)
# ============================================================

def run_tests():
    """
    运行 4 个测试演示，验证搜索引擎各核心模块的正确性。
    """

    # ---- 测试语料: 10 句精巧的英文短文 ----
    # 包含生僻词 "arachnocentric" 和常见词 "the", "a"
    documents = {
        1: "The quick brown fox jumps over a lazy dog in the park",
        2: "Stanford University is a prestigious research university located in Stanford California",
        3: "Information retrieval is the science of searching for information in a document",
        4: "The arachnocentric view of ecology focuses on the role of spiders in a web of life",
        5: "A search engine indexes documents and retrieves the most relevant results for a query",
        6: "The inverted index is a fundamental data structure used in search engines",
        7: "Natural language processing and information retrieval share many common techniques",
        8: "Spiders build intricate webs and the arachnocentric perspective studies their ecological impact",
        9: "The vector space model represents documents and queries as vectors in a high dimensional space",
        10: "Boolean retrieval uses logical operators like AND OR and NOT to find documents",
    }

    # 构建搜索引擎
    engine = MiniSearchEngine()
    engine.build(documents)

    # 展示索引样本
    engine.print_index_sample(max_terms=8)

    # ================================================================
    # 测试1: 布尔 AND 查询 (按 DF 优化)
    # ================================================================
    print("=" * 60)
    print("  测试1: 布尔 AND 查询 (Boolean AND Query)")
    print("  【演示: 按 DF 从小到大排序优化合并顺序】")
    print("=" * 60)

    query = "information retrieval"
    print(f"  查询: '{query}'")

    # 展示 DF 排序过程
    preprocessor = TextPreprocessor()
    terms = [t[0] for t in preprocessor.tokenize(query)]
    for term in terms:
        print(f"    term '{term}': DF = {engine.index.get_df(term)}")
    terms_sorted = sorted(terms, key=lambda t: engine.index.get_df(t))
    print(f"  DF 升序优化后的处理顺序: {terms_sorted}")

    results = engine.boolean_search(query)
    print(f"  AND 查询结果 (DocIDs): {results}")
    for doc_id in results:
        print(f"    Doc {doc_id}: \"{documents[doc_id]}\"")
    print()

    # ================================================================
    # 测试2: 短语查询 (Phrase Query)
    # ================================================================
    print("=" * 60)
    print("  测试2: 短语查询 (Phrase Query)")
    print("  【演示: 利用位置索引的位置偏移量检查】")
    print("=" * 60)

    phrase = "Stanford University"
    print(f"  短语查询: \"{phrase}\"")
    results = engine.phrase_search(phrase)
    print(f"  短语匹配结果 (DocIDs): {results}")
    for doc_id in results:
        print(f"    Doc {doc_id}: \"{documents[doc_id]}\"")
    print()

    phrase2 = "information retrieval"
    print(f"  短语查询: \"{phrase2}\"")
    results2 = engine.phrase_search(phrase2)
    print(f"  短语匹配结果 (DocIDs): {results2}")
    for doc_id in results2:
        print(f"    Doc {doc_id}: \"{documents[doc_id]}\"")
    print()

    # 反例: "retrieval information" 不应匹配（顺序不对）
    phrase3 = "retrieval information"
    print(f"  短语查询 (反例): \"{phrase3}\"")
    results3 = engine.phrase_search(phrase3)
    print(f"  短语匹配结果 (DocIDs): {results3}  ← 应为空（词序不匹配）")
    print()

    # ================================================================
    # 测试3: 拼写纠错 (Spelling Correction)
    # ================================================================
    print("=" * 60)
    print("  测试3: 拼写纠错 (Spelling Correction)")
    print("  【演示: 基于 Levenshtein 编辑距离的 Did you mean 提示】")
    print("=" * 60)

    misspelled_words = ["informaton", "retrival", "arachnocentrc", "univrsity", "srch"]
    for word in misspelled_words:
        suggestion = engine.spell_check(word)
        dist = SpellingCorrector.levenshtein_distance(word.lower(), suggestion)
        print(f"  '{word}' -> Did you mean '{suggestion}'? (编辑距离={dist})")
    print()

    # ================================================================
    # 测试4: 排序检索 (Ranked Retrieval)
    # ================================================================
    print("=" * 60)
    print("  测试4: 排序检索 (Ranked Retrieval - lnc.ltc)")
    print("  【演示: IDF 惩罚常见词，提权罕见词】")
    print("=" * 60)

    # 测试4a: 含罕见词的查询
    query_rare = "arachnocentric ecology spiders"
    print(f"\n  查询A (含罕见词): '{query_rare}'")
    print(f"  各查询词的 IDF 值:")
    N = engine.index.doc_count
    for t in ["arachnocentric", "ecology", "spiders"]:
        df = engine.index.get_df(t)
        if df > 0:
            idf_val = TFIDFScorer.idf(N, df)
            print(f"    '{t}': DF={df}, IDF={idf_val:.4f}  ← 罕见词 IDF 高")
        else:
            print(f"    '{t}': DF=0, IDF=N/A (不在词典中)")

    results_rare = engine.ranked_search(query_rare, k=5)
    print(f"  Top-5 结果:")
    for score, doc_id in results_rare:
        print(f"    Doc {doc_id} [score={score:.6f}]: \"{documents[doc_id]}\"")

    # 测试4b: 含常见词的查询
    query_common = "the a in"
    print(f"\n  查询B (全是常见词): '{query_common}'")
    print(f"  各查询词的 IDF 值:")
    for t in ["the", "a", "in"]:
        df = engine.index.get_df(t)
        if df > 0:
            idf_val = TFIDFScorer.idf(N, df)
            print(f"    '{t}': DF={df}, IDF={idf_val:.4f}  ← 常见词 IDF 低，权重被惩罚")

    results_common = engine.ranked_search(query_common, k=5)
    print(f"  Top-5 结果:")
    for score, doc_id in results_common:
        print(f"    Doc {doc_id} [score={score:.6f}]: \"{documents[doc_id]}\"")

    # 测试4c: 混合查询
    query_mix = "search engine information retrieval"
    print(f"\n  查询C (混合): '{query_mix}'")
    print(f"  各查询词的 IDF 值:")
    for t in ["search", "engine", "information", "retrieval"]:
        df = engine.index.get_df(t)
        if df > 0:
            idf_val = TFIDFScorer.idf(N, df)
            print(f"    '{t}': DF={df}, IDF={idf_val:.4f}")

    results_mix = engine.ranked_search(query_mix, k=5)
    print(f"  Top-5 结果:")
    for score, doc_id in results_mix:
        print(f"    Doc {doc_id} [score={score:.6f}]: \"{documents[doc_id]}\"")

    print()
    print("=" * 60)
    print("  所有测试完成！(All tests completed)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
