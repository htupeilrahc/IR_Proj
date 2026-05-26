"""
全功能微型文本搜索引擎 - 第二部分 (Mini Search Engine - Part 2)
================================================================
涵盖 IR 课程后半期内容：
- ch8:  检索评价 (Evaluation) — Precision, Recall, F1, MAP, NDCG, P@K
- ch9&11: 概率检索模型 (Probabilistic IR) — BM25, 语言模型 (LM)
- ch13&14: 文本分类 (Text Classification) — Naive Bayes, kNN, Rocchio
- ch15: 学习排序 (Learning to Rank) — 决策树, Pointwise LTR
- ch21: 链接分析 (Link Analysis) — PageRank, HITS
- personalization: 个性化检索 (Personalization) — 用户画像重排序

所有核心逻辑均使用 Python 原生数据结构手工实现，不依赖 sklearn 等框架。
"""

import math
import heapq
from collections import defaultdict, Counter
from mini_search_engine import (
    MiniSearchEngine, TextPreprocessor, TFIDFScorer,
    PositionalInvertedIndex, SPIMIIndexer
)


# ============================================================
# 模块8: 检索评价 (Evaluation Metrics)
# ============================================================

class EvaluationMetrics:
    """
    信息检索系统评价指标。

    对应教材第8章 (Evaluation in Information Retrieval)。
    实现了从基础到高级的完整评价指标体系：
    - 无序集合指标: Precision, Recall, F-measure
    - 有序列表指标: P@K, R-Precision, MAP, MRR
    - 带分级相关度指标: DCG, NDCG
    - Precision-Recall 曲线 (11-point interpolated)
    """

    @staticmethod
    def precision(retrieved: list, relevant: set) -> float:
        """
        精确率 (Precision): 检索到的文档中有多少是相关的。
        公式: P = |检索到 ∩ 相关| / |检索到|
        """
        if not retrieved:
            return 0.0
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(retrieved)

    @staticmethod
    def recall(retrieved: list, relevant: set) -> float:
        """
        召回率 (Recall): 相关文档中有多少被检索到。
        公式: R = |检索到 ∩ 相关| / |相关|
        """
        if not relevant:
            return 0.0
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(relevant)

    @staticmethod
    def f_measure(precision: float, recall: float, beta: float = 1.0) -> float:
        """
        F-measure: Precision 和 Recall 的加权调和平均。
        公式: F_β = (1+β²) * P * R / (β² * P + R)

        β=1 时即为 F1-score (同等重视 P 和 R)。
        β>1 偏重 Recall, β<1 偏重 Precision。
        """
        if precision + recall == 0:
            return 0.0
        return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    @staticmethod
    def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
        """
        P@K: 前 K 个结果中的精确率。
        只看排名列表的前 K 项，不考虑之后的结果。
        常用指标: P@5, P@10, P@20。
        """
        if k <= 0:
            return 0.0
        top_k = retrieved[:k]
        return sum(1 for doc in top_k if doc in relevant) / k

    @staticmethod
    def r_precision(retrieved: list, relevant: set) -> float:
        """
        R-Precision: 前 R 个结果中的精确率，其中 R = |relevant|。
        直觉: 如果有 R 个相关文档，看前 R 个结果中有多少是相关的。
        """
        r = len(relevant)
        if r == 0:
            return 0.0
        return EvaluationMetrics.precision_at_k(retrieved, relevant, r)

    @staticmethod
    def average_precision(retrieved: list, relevant: set) -> float:
        """
        Average Precision (AP): 单个查询的平均精确率。

        对每个相关文档出现的位置计算 P@K，然后取平均。
        公式: AP = (1/|relevant|) * Σ (P@k * rel(k))
        其中 rel(k) = 1 当第 k 个文档相关时。

        AP 综合考虑了排序质量和召回率。
        """
        if not relevant:
            return 0.0
        hits = 0
        sum_precision = 0.0
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                hits += 1
                # 在当前位置计算 Precision
                sum_precision += hits / (i + 1)
        return sum_precision / len(relevant)

    @staticmethod
    def mean_average_precision(queries_results: list) -> float:
        """
        MAP (Mean Average Precision): 多个查询的 AP 平均值。

        MAP 是 IR 系统整体性能最常用的单一数字指标。
        公式: MAP = (1/|Q|) * Σ AP(q) for q in Q

        参数:
            queries_results: [(retrieved_list, relevant_set), ...]
        """
        if not queries_results:
            return 0.0
        total_ap = sum(
            EvaluationMetrics.average_precision(retrieved, relevant)
            for retrieved, relevant in queries_results
        )
        return total_ap / len(queries_results)

    @staticmethod
    def reciprocal_rank(retrieved: list, relevant: set) -> float:
        """
        Reciprocal Rank (RR): 第一个相关文档的排名倒数。
        公式: RR = 1 / rank_of_first_relevant
        MRR = 多个查询的 RR 平均值。
        """
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def dcg_at_k(rankings: list, k: int) -> float:
        """
        DCG@K (Discounted Cumulative Gain): 带折扣的累积增益。

        适用于多级相关度 (graded relevance)，如: 0=不相关, 1=部分相关, 2=高度相关。
        公式: DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log2(i+1)

        核心思想: 排名靠前的位置价值更高（对数折扣），相关度越高增益越大。

        参数:
            rankings: [rel_score_1, rel_score_2, ...] 按检索排序的相关度分数列表
        """
        dcg = 0.0
        for i in range(min(k, len(rankings))):
            rel = rankings[i]
            # (2^rel - 1) / log2(i+2) — 注意 i 是从 0 开始，所以分母用 i+2
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        return dcg

    @staticmethod
    def ndcg_at_k(rankings: list, k: int) -> float:
        """
        NDCG@K (Normalized DCG): 归一化的 DCG。

        将 DCG 除以"理想排序"的 DCG (IDCG)，使得指标在 [0,1] 之间。
        公式: NDCG@K = DCG@K / IDCG@K

        IDCG 是将所有文档按相关度降序排列后的 DCG（最优情况）。
        """
        dcg = EvaluationMetrics.dcg_at_k(rankings, k)
        # 计算 Ideal DCG: 按相关度降序排列
        ideal_rankings = sorted(rankings, reverse=True)
        idcg = EvaluationMetrics.dcg_at_k(ideal_rankings, k)
        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def interpolated_precision_recall(retrieved: list, relevant: set) -> dict:
        """
        11 点插值 Precision-Recall 曲线。

        在 11 个标准召回率水平 (0.0, 0.1, ..., 1.0) 处计算插值精确率。
        插值规则: P_interp(r) = max(P(r')) for all r' >= r

        这是绘制 P-R 曲线的标准方法，用于对比不同系统的性能。
        """
        # 计算每个位置的 precision 和 recall
        precisions = []
        recalls = []
        hits = 0
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                hits += 1
            p = hits / (i + 1)
            r = hits / len(relevant) if relevant else 0
            precisions.append(p)
            recalls.append(r)

        # 11 点插值
        recall_levels = [i * 0.1 for i in range(11)]
        interpolated = {}
        for level in recall_levels:
            # 找到 recall >= level 的所有位置中的最大 precision
            max_p = 0.0
            for i in range(len(recalls)):
                if recalls[i] >= level:
                    max_p = max(max_p, precisions[i])
            interpolated[round(level, 1)] = round(max_p, 4)
        return interpolated


# ============================================================
# 模块9&11: 概率检索模型 (Probabilistic IR)
# ============================================================

class BM25Scorer:
    """
    Okapi BM25 评分模型。

    对应教材第11章 (Probabilistic Information Retrieval)。
    BM25 是概率检索的最经典实现，在实际搜索引擎中广泛使用。

    BM25 评分公式:
    score(D, Q) = Σ_{t∈Q} IDF(t) * [tf(t,D) * (k1+1)] / [tf(t,D) + k1*(1 - b + b*|D|/avgdl)]

    其中:
    - k1: TF 饱和参数 (典型值 1.2~2.0)，控制 TF 增长速度
    - b:  文档长度归一化参数 (典型值 0.75)，b=0 不归一化，b=1 完全归一化
    - |D|: 文档长度
    - avgdl: 平均文档长度

    相比 TF-IDF, BM25 的核心改进:
    1. TF 有饱和效应 (saturation): TF 增大时权重增长逐渐平缓，避免长文档垄断
    2. 文档长度归一化: 通过 b 参数灵活控制长文档的惩罚程度
    3. IDF 使用概率公式，对极高频词可产生负权重（通常截断为0）
    """

    def __init__(self, index: PositionalInvertedIndex, k1: float = 1.5, b: float = 0.75):
        """
        参数:
            index: 位置倒排索引
            k1: TF 饱和参数（默认 1.5）
            b: 文档长度归一化参数（默认 0.75）
        """
        self.index = index
        self.k1 = k1
        self.b = b
        self.preprocessor = TextPreprocessor()
        # 预计算: 每个文档的长度 (token 数)
        self.doc_lengths = {}
        self.avgdl = 0.0
        self._compute_doc_lengths()

    def _compute_doc_lengths(self):
        """预计算所有文档长度和平均文档长度"""
        total_length = 0
        for doc_id, text in self.index.documents.items():
            tokens = self.preprocessor.tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
        if self.index.doc_count > 0:
            self.avgdl = total_length / self.index.doc_count

    def idf_bm25(self, term: str) -> float:
        """
        BM25 的 IDF 公式 (Robertson-Sparck Jones 权重):
        IDF(t) = log((N - df + 0.5) / (df + 0.5))

        当 df > N/2 时 IDF 为负，表示该词太常见无区分力。
        实际系统中通常将负值截断为 0 或使用一个小的正数。
        """
        N = self.index.doc_count
        df = self.index.get_df(term)
        if df == 0:
            return 0.0
        idf = math.log((N - df + 0.5) / (df + 0.5))
        return max(idf, 0.0)  # 截断负值

    def score_document(self, query_terms: list, doc_id: int) -> float:
        """
        计算单个文档的 BM25 分数。

        BM25(D, Q) = Σ IDF(t) * [tf * (k1+1)] / [tf + k1*(1-b+b*|D|/avgdl)]
        """
        score = 0.0
        doc_len = self.doc_lengths.get(doc_id, 0)

        for term in query_terms:
            tf = self.index.get_tf(term, doc_id)
            if tf == 0:
                continue

            idf = self.idf_bm25(term)

            # BM25 TF 归一化: 引入文档长度归一化和 TF 饱和
            # 分子: tf * (k1 + 1)
            numerator = tf * (self.k1 + 1)
            # 分母: tf + k1 * (1 - b + b * |D| / avgdl)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, k: int = 10) -> list:
        """
        BM25 排序检索，返回 Top-K 结果。
        使用最小堆 (Min-Heap) 提取 Top-K。
        """
        tokens = self.preprocessor.tokenize(query)
        query_terms = [t[0] for t in tokens]

        if not query_terms:
            return []

        # 收集候选文档: 至少包含一个查询词的文档
        candidate_docs = set()
        for term in query_terms:
            for doc_id in self.index.get_postings(term).keys():
                candidate_docs.add(doc_id)

        # 打分并用最小堆维护 Top-K
        min_heap = []
        for doc_id in candidate_docs:
            score = self.score_document(query_terms, doc_id)
            if score > 0:
                if len(min_heap) < k:
                    heapq.heappush(min_heap, (score, doc_id))
                elif score > min_heap[0][0]:
                    heapq.heapreplace(min_heap, (score, doc_id))

        return sorted(min_heap, key=lambda x: -x[0])


class LanguageModelScorer:
    """
    统计语言模型检索 (Language Model for IR)。

    对应教材第12章概念。核心思想:
    - 为每个文档建立一个一元语言模型 (Unigram LM)
    - 查询的得分 = 查询在文档语言模型下的生成概率 P(Q|D)
    - 使用对数概率避免下溢: log P(Q|D) = Σ log P(t|D)

    平滑 (Smoothing) 是语言模型的关键:
    - 未在文档中出现的词不能概率为 0
    - 需要将部分概率质量分配给未见词

    本实现提供两种经典平滑方法:
    1. Jelinek-Mercer (线性插值) 平滑: P(t|D) = λ·P_ml(t|D) + (1-λ)·P(t|C)
    2. Dirichlet (贝叶斯) 平滑: P(t|D) = (tf + μ·P(t|C)) / (|D| + μ)
    """

    def __init__(self, index: PositionalInvertedIndex, method: str = "dirichlet",
                 lambda_param: float = 0.7, mu: float = 2000):
        """
        参数:
            index: 位置倒排索引
            method: "jelinek-mercer" 或 "dirichlet"
            lambda_param: JM 平滑的 λ 参数 (0~1)，λ 越大越信任文档
            mu: Dirichlet 平滑的 μ 参数，μ 越大越信任集合
        """
        self.index = index
        self.method = method
        self.lambda_param = lambda_param
        self.mu = mu
        self.preprocessor = TextPreprocessor()

        # 预计算集合语言模型 P(t|C)
        self.collection_prob = {}
        self.collection_length = 0
        self.doc_lengths = {}
        self._build_collection_model()

    def _build_collection_model(self):
        """
        构建集合语言模型 (Collection Language Model)。
        P(t|C) = cf(t) / |C|
        其中 cf(t) 是 term 在整个集合中的总出现次数。
        """
        total_tokens = 0
        term_cf = defaultdict(int)  # Collection Frequency

        for doc_id, text in self.index.documents.items():
            tokens = self.preprocessor.tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_tokens += len(tokens)
            for term, _ in tokens:
                term_cf[term] += 1

        self.collection_length = total_tokens
        for term, cf in term_cf.items():
            self.collection_prob[term] = cf / total_tokens

    def score_document(self, query_terms: list, doc_id: int) -> float:
        """
        计算 log P(Q|M_D) — 查询在文档语言模型下的对数概率。
        """
        log_score = 0.0
        doc_len = self.doc_lengths.get(doc_id, 0)

        for term in query_terms:
            tf = self.index.get_tf(term, doc_id)
            p_collection = self.collection_prob.get(term, 1e-10)

            if self.method == "jelinek-mercer":
                # Jelinek-Mercer 平滑:
                # P(t|D) = λ * (tf/|D|) + (1-λ) * P(t|C)
                p_ml = tf / doc_len if doc_len > 0 else 0
                p_smooth = self.lambda_param * p_ml + (1 - self.lambda_param) * p_collection
            else:
                # Dirichlet 平滑:
                # P(t|D) = (tf + μ * P(t|C)) / (|D| + μ)
                p_smooth = (tf + self.mu * p_collection) / (doc_len + self.mu)

            if p_smooth > 0:
                log_score += math.log(p_smooth)
            else:
                log_score += -100  # 极小概率

        return log_score

    def search(self, query: str, k: int = 10) -> list:
        """语言模型排序检索"""
        tokens = self.preprocessor.tokenize(query)
        query_terms = [t[0] for t in tokens]

        if not query_terms:
            return []

        # 对所有文档打分（语言模型不需要候选集优化，因为平滑使所有文档都有非零概率）
        min_heap = []
        for doc_id in self.index.documents:
            score = self.score_document(query_terms, doc_id)
            if len(min_heap) < k:
                heapq.heappush(min_heap, (score, doc_id))
            elif score > min_heap[0][0]:
                heapq.heapreplace(min_heap, (score, doc_id))

        return sorted(min_heap, key=lambda x: -x[0])


# ============================================================
# 模块13&14: 文本分类 (Text Classification)
# ============================================================

class NaiveBayesClassifier:
    """
    多项式朴素贝叶斯分类器 (Multinomial Naive Bayes)。

    对应教材第13章 (Text Classification and Naive Bayes)。

    核心思想 (贝叶斯定理):
    P(c|d) ∝ P(c) * P(d|c)
            = P(c) * Π P(t|c)^{tf(t,d)}

    取对数避免下溢:
    log P(c|d) ∝ log P(c) + Σ tf(t,d) * log P(t|c)

    关键技术:
    - 拉普拉斯平滑 (Laplace/Add-1 Smoothing): 避免零概率问题
      P(t|c) = (count(t,c) + 1) / (Σ count(t',c) + |V|)
    - 先验概率 P(c) = N_c / N
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        # 类别先验: log P(c)
        self.log_prior = {}
        # 条件概率: log P(t|c) for each class
        self.log_likelihood = defaultdict(dict)
        # 词典
        self.vocabulary = set()
        # 类别列表
        self.classes = []

    def train(self, documents: list, labels: list):
        """
        训练朴素贝叶斯分类器。

        参数:
            documents: [text1, text2, ...]
            labels: [class1, class2, ...] 对应类别标签
        """
        self.classes = list(set(labels))
        N = len(documents)

        # 统计每个类别的文档数和词频
        class_doc_count = Counter(labels)
        class_word_count = defaultdict(lambda: defaultdict(int))
        class_total_words = defaultdict(int)

        for doc, label in zip(documents, labels):
            tokens = self.preprocessor.tokenize(doc)
            for term, _ in tokens:
                self.vocabulary.add(term)
                class_word_count[label][term] += 1
                class_total_words[label] += 1

        V = len(self.vocabulary)

        # 计算 log 先验概率: log P(c) = log(N_c / N)
        for c in self.classes:
            self.log_prior[c] = math.log(class_doc_count[c] / N)

        # 计算 log 条件概率 (带拉普拉斯平滑):
        # log P(t|c) = log((count(t,c) + 1) / (total_words_in_c + |V|))
        for c in self.classes:
            total = class_total_words[c]
            for term in self.vocabulary:
                count = class_word_count[c].get(term, 0)
                # 拉普拉斯平滑 (Add-1 Smoothing)
                self.log_likelihood[c][term] = math.log((count + 1) / (total + V))

    def predict(self, document: str) -> tuple:
        """
        预测文档类别。

        返回: (predicted_class, {class: score}) 最可能的类别及各类别得分
        """
        tokens = self.preprocessor.tokenize(document)
        scores = {}

        for c in self.classes:
            # score = log P(c) + Σ tf * log P(t|c)
            score = self.log_prior[c]
            for term, _ in tokens:
                if term in self.log_likelihood[c]:
                    score += self.log_likelihood[c][term]
                else:
                    # 未知词使用平滑: log(1 / (total + |V|))
                    score += math.log(1 / (len(self.vocabulary) + len(self.vocabulary)))
            scores[c] = score

        predicted = max(scores, key=scores.get)
        return predicted, scores


class KNNClassifier:
    """
    K 近邻分类器 (K-Nearest Neighbors, kNN)。

    对应教材第14章 (Vector Space Classification)。

    核心思想:
    - 将文档表示为 TF-IDF 向量
    - 找到待分类文档的 K 个最近邻（余弦相似度最大）
    - 以 K 个邻居中的多数类别作为预测结果

    优点: 简单有效，无需训练阶段
    缺点: 分类时计算量大（需与所有训练文档比较）
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.preprocessor = TextPreprocessor()
        self.train_docs = []  # [(tfidf_vector, label), ...]
        self.vocabulary = {}  # term -> index
        self.idf = {}  # term -> idf value
        self.N = 0

    def _build_vocab_and_idf(self, documents: list):
        """构建词典和 IDF"""
        df = defaultdict(int)
        for doc in documents:
            tokens = self.preprocessor.tokenize(doc)
            seen = set()
            for term, _ in tokens:
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
                if term not in seen:
                    df[term] += 1
                    seen.add(term)
        self.N = len(documents)
        for term, freq in df.items():
            self.idf[term] = math.log10(self.N / freq) if freq > 0 else 0

    def _vectorize(self, text: str) -> dict:
        """将文本转为 TF-IDF 向量 (稀疏表示)"""
        tokens = self.preprocessor.tokenize(text)
        tf = defaultdict(int)
        for term, _ in tokens:
            tf[term] += 1
        # TF-IDF
        vector = {}
        for term, freq in tf.items():
            if term in self.idf:
                tfidf = (1 + math.log10(freq)) * self.idf[term]
                vector[term] = tfidf
        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            vector = {t: v / norm for t, v in vector.items()}
        return vector

    @staticmethod
    def _cosine_similarity(v1: dict, v2: dict) -> float:
        """计算两个稀疏向量的余弦相似度（已归一化则为点积）"""
        score = 0.0
        for term, weight in v1.items():
            if term in v2:
                score += weight * v2[term]
        return score

    def train(self, documents: list, labels: list):
        """训练: 存储所有训练文档的 TF-IDF 向量"""
        self._build_vocab_and_idf(documents)
        self.train_docs = []
        for doc, label in zip(documents, labels):
            vec = self._vectorize(doc)
            self.train_docs.append((vec, label))

    def predict(self, document: str) -> tuple:
        """
        预测: 找 K 个最近邻，多数投票。
        返回: (predicted_class, [(similarity, label), ...] top-K 邻居)
        """
        query_vec = self._vectorize(document)

        # 计算与所有训练文档的相似度
        similarities = []
        for vec, label in self.train_docs:
            sim = self._cosine_similarity(query_vec, vec)
            similarities.append((sim, label))

        # 取 Top-K
        similarities.sort(key=lambda x: -x[0])
        top_k = similarities[:self.k]

        # 多数投票
        votes = Counter(label for _, label in top_k)
        predicted = votes.most_common(1)[0][0]
        return predicted, top_k


class RocchioClassifier:
    """
    Rocchio 分类器 (基于质心的向量空间分类)。

    对应教材第14章。

    核心思想:
    - 为每个类别计算其文档向量的质心 (centroid)
    - 分类时将文档分配给最近的质心

    质心计算: μ(c) = (1/|D_c|) * Σ v(d) for d in D_c
    分类决策: argmax_c cosine(v(d), μ(c))
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.centroids = {}  # class -> centroid vector
        self.vocabulary = {}
        self.idf = {}
        self.N = 0

    def _build_vocab_and_idf(self, documents: list):
        """构建词典和 IDF"""
        df = defaultdict(int)
        for doc in documents:
            tokens = self.preprocessor.tokenize(doc)
            seen = set()
            for term, _ in tokens:
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
                if term not in seen:
                    df[term] += 1
                    seen.add(term)
        self.N = len(documents)
        for term, freq in df.items():
            self.idf[term] = math.log10(self.N / freq) if freq > 0 else 0

    def _vectorize(self, text: str) -> dict:
        """TF-IDF 向量化"""
        tokens = self.preprocessor.tokenize(text)
        tf = defaultdict(int)
        for term, _ in tokens:
            tf[term] += 1
        vector = {}
        for term, freq in tf.items():
            if term in self.idf:
                vector[term] = (1 + math.log10(freq)) * self.idf[term]
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            vector = {t: v / norm for t, v in vector.items()}
        return vector

    def train(self, documents: list, labels: list):
        """训练: 计算每个类别的质心向量"""
        self._build_vocab_and_idf(documents)
        class_vectors = defaultdict(list)

        for doc, label in zip(documents, labels):
            vec = self._vectorize(doc)
            class_vectors[label].append(vec)

        # 计算质心: 各维度取均值
        for label, vectors in class_vectors.items():
            centroid = defaultdict(float)
            for vec in vectors:
                for term, weight in vec.items():
                    centroid[term] += weight
            n = len(vectors)
            centroid = {t: w / n for t, w in centroid.items()}
            # 归一化质心
            norm = math.sqrt(sum(v * v for v in centroid.values()))
            if norm > 0:
                centroid = {t: v / norm for t, v in centroid.items()}
            self.centroids[label] = centroid

    def predict(self, document: str) -> tuple:
        """分类: 找最近质心"""
        vec = self._vectorize(document)
        scores = {}
        for label, centroid in self.centroids.items():
            sim = sum(vec.get(t, 0) * w for t, w in centroid.items())
            scores[label] = sim
        predicted = max(scores, key=scores.get)
        return predicted, scores


# ============================================================
# 模块15: 学习排序 (Learning to Rank)
# ============================================================

class DecisionTreeNode:
    """决策树节点"""
    def __init__(self):
        self.feature_index = None  # 分裂特征索引
        self.threshold = None      # 分裂阈值
        self.left = None           # 左子树 (<=threshold)
        self.right = None          # 右子树 (>threshold)
        self.value = None          # 叶子节点的预测值
        self.is_leaf = False


class DecisionTreeRanker:
    """
    基于决策树的学习排序 (Learning to Rank)。

    对应教材第15章 (Learning to Rank)。

    学习排序 (LTR) 的核心思想:
    - 将排序问题转化为机器学习问题
    - 为每个 (查询, 文档) 对提取特征
    - 用训练数据学习排序函数

    三大方法:
    1. Pointwise: 将排序转为回归/分类，独立预测每个文档的相关度
    2. Pairwise: 学习文档对之间的偏序关系（如 RankSVM, LambdaMART）
    3. Listwise: 直接优化整个排序列表（如 ListNet）

    本实现采用 Pointwise 方法 + 简单决策树:
    - 特征: BM25 分数、TF-IDF 分数、文档长度、查询-文档重叠度等
    - 用决策树回归器预测相关度分数
    """

    def __init__(self, max_depth: int = 5, min_samples: int = 2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.feature_names = [
            "bm25_score",       # BM25 评分
            "tfidf_score",      # TF-IDF 余弦相似度
            "query_term_ratio", # 查询词在文档中出现的比例
            "doc_length",       # 文档长度 (归一化)
            "avg_tf",           # 查询词在文档中的平均 TF
            "idf_sum",          # 查询词 IDF 之和
        ]

    def _mse(self, values: list) -> float:
        """计算均方误差 (用于回归树的分裂准则)"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _find_best_split(self, X: list, y: list) -> tuple:
        """
        寻找最佳分裂点 (特征 + 阈值)。
        遍历所有特征和所有可能的阈值，选择使 MSE 降低最多的分裂。
        """
        best_feature = None
        best_threshold = None
        best_mse_reduction = 0.0
        current_mse = self._mse(y)
        n = len(y)

        num_features = len(X[0]) if X else 0

        for feature_idx in range(num_features):
            # 获取该特征的所有值并排序去重
            values = sorted(set(x[feature_idx] for x in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2

                # 分裂
                left_y = [y[j] for j in range(n) if X[j][feature_idx] <= threshold]
                right_y = [y[j] for j in range(n) if X[j][feature_idx] > threshold]

                if len(left_y) < self.min_samples or len(right_y) < self.min_samples:
                    continue

                # 加权 MSE
                weighted_mse = (len(left_y) * self._mse(left_y) +
                                len(right_y) * self._mse(right_y)) / n
                mse_reduction = current_mse - weighted_mse

                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X: list, y: list, depth: int) -> DecisionTreeNode:
        """递归构建决策树"""
        node = DecisionTreeNode()

        # 终止条件: 达到最大深度或样本数不足
        if depth >= self.max_depth or len(y) <= self.min_samples:
            node.is_leaf = True
            node.value = sum(y) / len(y) if y else 0.0
            return node

        # 所有标签相同
        if len(set(y)) == 1:
            node.is_leaf = True
            node.value = y[0]
            return node

        # 寻找最佳分裂
        feature_idx, threshold = self._find_best_split(X, y)

        if feature_idx is None:
            node.is_leaf = True
            node.value = sum(y) / len(y)
            return node

        node.feature_index = feature_idx
        node.threshold = threshold

        # 分裂数据
        left_indices = [i for i in range(len(X)) if X[i][feature_idx] <= threshold]
        right_indices = [i for i in range(len(X)) if X[i][feature_idx] > threshold]

        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]

        node.left = self._build_tree(left_X, left_y, depth + 1)
        node.right = self._build_tree(right_X, right_y, depth + 1)

        return node

    def train(self, X: list, y: list):
        """
        训练决策树排序器。
        X: [[feature1, feature2, ...], ...] 特征矩阵
        y: [relevance_score, ...] 相关度标签
        """
        self.root = self._build_tree(X, y, 0)

    def predict_one(self, features: list) -> float:
        """预测单个样本的相关度分数"""
        node = self.root
        while not node.is_leaf:
            if features[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X: list) -> list:
        """批量预测"""
        return [self.predict_one(x) for x in X]


class LearningToRank:
    """
    学习排序系统: 整合特征提取 + 决策树排序。

    特征设计 (Feature Engineering):
    - BM25 分数: 概率检索信号
    - TF-IDF 余弦分数: 向量空间模型信号
    - 查询词覆盖率: 查询词在文档中出现的比例
    - 文档长度: 归一化后的文档长度
    - 平均 TF: 查询词在文档中的平均词频
    - IDF 总和: 查询中匹配词的 IDF 累计
    """

    def __init__(self, index: PositionalInvertedIndex):
        self.index = index
        self.preprocessor = TextPreprocessor()
        self.bm25 = BM25Scorer(index)
        self.ranker = DecisionTreeRanker(max_depth=4, min_samples=2)
        self.max_doc_length = max(
            len(self.preprocessor.tokenize(text))
            for text in index.documents.values()
        ) if index.documents else 1

    def extract_features(self, query: str, doc_id: int) -> list:
        """
        为 (query, doc) 对提取特征向量。
        """
        tokens = self.preprocessor.tokenize(query)
        query_terms = [t[0] for t in tokens]
        N = self.index.doc_count

        # Feature 1: BM25 分数
        bm25_score = self.bm25.score_document(query_terms, doc_id)

        # Feature 2: TF-IDF 余弦相似度 (简化版)
        tfidf_score = 0.0
        doc_norm = self.index.doc_norms.get(doc_id, 1.0)
        for term in query_terms:
            tf = self.index.get_tf(term, doc_id)
            df = self.index.get_df(term)
            if tf > 0 and df > 0:
                doc_w = (1 + math.log10(tf)) / doc_norm if doc_norm > 0 else 0
                query_w = (1 + math.log10(1)) * math.log10(N / df)
                tfidf_score += doc_w * query_w

        # Feature 3: 查询词覆盖率
        matched_terms = sum(1 for t in query_terms if self.index.get_tf(t, doc_id) > 0)
        query_term_ratio = matched_terms / len(query_terms) if query_terms else 0

        # Feature 4: 文档长度 (归一化)
        doc_text = self.index.documents.get(doc_id, "")
        doc_length = len(self.preprocessor.tokenize(doc_text)) / self.max_doc_length

        # Feature 5: 平均 TF
        total_tf = sum(self.index.get_tf(t, doc_id) for t in query_terms)
        avg_tf = total_tf / len(query_terms) if query_terms else 0

        # Feature 6: 匹配词的 IDF 之和
        idf_sum = sum(
            math.log10(N / self.index.get_df(t))
            for t in query_terms
            if self.index.get_df(t) > 0 and self.index.get_tf(t, doc_id) > 0
        )

        return [bm25_score, tfidf_score, query_term_ratio, doc_length, avg_tf, idf_sum]

    def train(self, training_data: list):
        """
        训练排序模型。
        training_data: [(query, doc_id, relevance_score), ...]
        """
        X = []
        y = []
        for query, doc_id, relevance in training_data:
            features = self.extract_features(query, doc_id)
            X.append(features)
            y.append(relevance)
        self.ranker.train(X, y)

    def rank(self, query: str, candidate_docs: list, k: int = 5) -> list:
        """
        用学习排序模型对候选文档排序。
        返回: [(predicted_score, doc_id), ...] 按分数降序
        """
        scored = []
        for doc_id in candidate_docs:
            features = self.extract_features(query, doc_id)
            score = self.ranker.predict_one(features)
            scored.append((score, doc_id))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


# ============================================================
# 模块21: 链接分析 (Link Analysis)
# ============================================================

class PageRank:
    """
    PageRank 算法。

    对应教材第21章 (Link Analysis)。

    核心思想 (Random Surfer Model):
    - 一个随机网络冲浪者，在每一步:
      * 以概率 d (damping factor) 点击当前页面的一个出链
      * 以概率 (1-d) 随机跳转到任意页面
    - PageRank 值 = 该冲浪者最终访问各页面的稳态概率

    迭代公式:
    PR(p) = (1-d)/N + d * Σ PR(q)/L(q) for all q that link to p

    其中:
    - d: 阻尼因子 (damping factor)，通常取 0.85
    - N: 总页面数
    - L(q): 页面 q 的出链数

    收敛条件: ||PR^{k+1} - PR^k|| < ε
    """

    def __init__(self, damping: float = 0.85, max_iter: int = 100, epsilon: float = 1e-6):
        self.damping = damping
        self.max_iter = max_iter
        self.epsilon = epsilon

    def compute(self, graph: dict) -> dict:
        """
        计算 PageRank。

        参数:
            graph: {node: [outgoing_links], ...} 邻接表

        返回:
            {node: pagerank_score, ...}
        """
        # 收集所有节点
        nodes = set(graph.keys())
        for targets in graph.values():
            for t in targets:
                nodes.add(t)
        nodes = list(nodes)
        N = len(nodes)

        # 初始化: 每个节点均匀分配
        pr = {node: 1.0 / N for node in nodes}

        # 计算入链关系
        in_links = defaultdict(list)
        out_degree = {}
        for node in nodes:
            out_links = graph.get(node, [])
            out_degree[node] = len(out_links)
            for target in out_links:
                in_links[target].append(node)

        # 迭代计算
        iterations = 0
        history = [dict(pr)]  # 记录迭代历史

        for _ in range(self.max_iter):
            new_pr = {}
            for node in nodes:
                # 来自入链的贡献
                rank_sum = 0.0
                for src in in_links[node]:
                    if out_degree[src] > 0:
                        rank_sum += pr[src] / out_degree[src]

                # PageRank 公式
                new_pr[node] = (1 - self.damping) / N + self.damping * rank_sum

            # 检查收敛
            diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
            pr = new_pr
            iterations += 1
            history.append(dict(pr))

            if diff < self.epsilon:
                break

        return pr, iterations, history


class HITS:
    """
    HITS 算法 (Hyperlink-Induced Topic Search)。

    对应教材第21章。又称 Hubs and Authorities 算法。

    核心思想:
    - 好的 Hub 页面指向很多好的 Authority 页面
    - 好的 Authority 页面被很多好的 Hub 页面指向
    - 这是一个相互增强 (mutual reinforcement) 的关系

    迭代更新规则:
    - Authority 更新: auth(p) = Σ hub(q) for all q that link to p
    - Hub 更新: hub(p) = Σ auth(q) for all q that p links to

    每次迭代后需要归一化 (L2 norm)。

    与 PageRank 的区别:
    - PageRank 是查询无关的 (query-independent)，离线计算
    - HITS 是查询相关的 (query-dependent)，在线针对检索结果集计算
    """

    def __init__(self, max_iter: int = 50, epsilon: float = 1e-6):
        self.max_iter = max_iter
        self.epsilon = epsilon

    def compute(self, graph: dict) -> tuple:
        """
        计算 Hub 和 Authority 值。

        参数:
            graph: {node: [outgoing_links], ...}

        返回:
            (hub_scores, authority_scores, iterations)
        """
        # 收集所有节点
        nodes = set(graph.keys())
        for targets in graph.values():
            for t in targets:
                nodes.add(t)
        nodes = list(nodes)

        # 初始化
        hub = {n: 1.0 for n in nodes}
        auth = {n: 1.0 for n in nodes}

        # 计算入链关系
        in_links = defaultdict(list)
        for node, targets in graph.items():
            for target in targets:
                in_links[target].append(node)

        iterations = 0
        for _ in range(self.max_iter):
            # Authority 更新: auth(p) = Σ hub(q) for q -> p
            new_auth = {}
            for node in nodes:
                new_auth[node] = sum(hub.get(src, 0) for src in in_links.get(node, []))

            # Hub 更新: hub(p) = Σ auth(q) for p -> q
            new_hub = {}
            for node in nodes:
                targets = graph.get(node, [])
                new_hub[node] = sum(new_auth.get(t, 0) for t in targets)

            # L2 归一化
            auth_norm = math.sqrt(sum(v * v for v in new_auth.values()))
            hub_norm = math.sqrt(sum(v * v for v in new_hub.values()))

            if auth_norm > 0:
                new_auth = {n: v / auth_norm for n, v in new_auth.items()}
            if hub_norm > 0:
                new_hub = {n: v / hub_norm for n, v in new_hub.items()}

            # 检查收敛
            auth_diff = sum(abs(new_auth[n] - auth[n]) for n in nodes)
            hub_diff = sum(abs(new_hub[n] - hub[n]) for n in nodes)

            auth = new_auth
            hub = new_hub
            iterations += 1

            if auth_diff < self.epsilon and hub_diff < self.epsilon:
                break

        return hub, auth, iterations


# ============================================================
# 个性化模块 (Personalization)
# ============================================================

class SearchPersonalizer:
    """
    搜索个性化 (Search Personalization)。

    对应 personalization 课件。

    核心思想:
    - 不同用户对同一查询有不同的信息需求
    - 通过用户画像 (User Profile) 对搜索结果进行个性化重排序
    - 用户画像可基于: 历史点击、浏览行为、兴趣标签等

    本实现:
    1. 用户画像构建: 基于用户历史点击文档，构建兴趣向量
    2. 个性化重排序: 将原始检索分数与用户兴趣相似度加权融合

    融合公式:
    final_score = α * retrieval_score + (1-α) * personalization_score

    其中 α 控制原始排序与个性化的平衡。
    """

    def __init__(self, index: PositionalInvertedIndex, alpha: float = 0.7):
        """
        参数:
            alpha: 原始检索分数的权重 (0~1)。
                   α=1.0 表示完全不个性化；α=0.0 表示完全依赖用户兴趣。
        """
        self.index = index
        self.alpha = alpha
        self.preprocessor = TextPreprocessor()
        self.user_profiles = {}  # user_id -> interest vector

    def update_profile(self, user_id: str, clicked_doc_ids: list):
        """
        基于用户点击历史更新用户兴趣画像。

        用户画像 = 点击文档的 TF-IDF 向量的质心 (归一化)。
        """
        profile = defaultdict(float)
        count = 0

        for doc_id in clicked_doc_ids:
            text = self.index.documents.get(doc_id, "")
            if not text:
                continue
            tokens = self.preprocessor.tokenize(text)
            tf = defaultdict(int)
            for term, _ in tokens:
                tf[term] += 1
            for term, freq in tf.items():
                df = self.index.get_df(term)
                if df > 0:
                    # 使用 log TF * IDF 作为权重
                    weight = (1 + math.log10(freq)) * math.log10(self.index.doc_count / df)
                    profile[term] += weight
            count += 1

        # 取均值并归一化
        if count > 0:
            profile = {t: w / count for t, w in profile.items()}
            norm = math.sqrt(sum(w * w for w in profile.values()))
            if norm > 0:
                profile = {t: w / norm for t, w in profile.items()}

        self.user_profiles[user_id] = dict(profile)

    def personalize(self, user_id: str, ranked_results: list) -> list:
        """
        对检索结果进行个性化重排序。

        参数:
            ranked_results: [(score, doc_id), ...] 原始排序结果
        返回:
            [(final_score, doc_id, original_score, personal_score), ...] 重排后的结果
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            # 无用户画像，返回原始排序
            return [(s, d, s, 0.0) for s, d in ranked_results]

        # 归一化原始分数到 [0, 1]
        max_score = max(s for s, _ in ranked_results) if ranked_results else 1.0
        min_score = min(s for s, _ in ranked_results) if ranked_results else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        reranked = []
        for orig_score, doc_id in ranked_results:
            # 计算文档与用户画像的相似度
            text = self.index.documents.get(doc_id, "")
            tokens = self.preprocessor.tokenize(text)
            tf = defaultdict(int)
            for term, _ in tokens:
                tf[term] += 1

            doc_vec = {}
            for term, freq in tf.items():
                df = self.index.get_df(term)
                if df > 0:
                    doc_vec[term] = (1 + math.log10(freq)) * math.log10(self.index.doc_count / df)
            # 归一化
            norm = math.sqrt(sum(w * w for w in doc_vec.values()))
            if norm > 0:
                doc_vec = {t: w / norm for t, w in doc_vec.items()}

            # 余弦相似度 (个性化分数)
            personal_score = sum(
                doc_vec.get(t, 0) * w for t, w in profile.items()
            )

            # 融合: α * normalized_retrieval + (1-α) * personalization
            norm_orig = (orig_score - min_score) / score_range
            final_score = self.alpha * norm_orig + (1 - self.alpha) * personal_score

            reranked.append((final_score, doc_id, orig_score, personal_score))

        reranked.sort(key=lambda x: -x[0])
        return reranked


# ============================================================
# 测试用例 (Test Cases)
# ============================================================

def run_tests():
    """运行所有后半期模块的测试演示。"""

    # ---- 共用测试语料 ----
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

    # 构建索引
    engine = MiniSearchEngine()
    engine.build(documents)

    # ================================================================
    # 测试1: 检索评价 (Evaluation)
    # ================================================================
    print("=" * 70)
    print("  测试1: 检索评价指标 (Evaluation Metrics)")
    print("  【ch8: Precision, Recall, MAP, NDCG】")
    print("=" * 70)

    # 模拟: 查询 "information retrieval" 的相关文档
    relevant = {3, 7}  # 真实相关的文档
    retrieved = [3, 5, 7, 6, 10]  # 系统返回的排序列表

    print(f"  相关文档集合 (ground truth): {relevant}")
    print(f"  系统返回列表 (ranked): {retrieved}")
    print()

    p = EvaluationMetrics.precision(retrieved, relevant)
    r = EvaluationMetrics.recall(retrieved, relevant)
    f1 = EvaluationMetrics.f_measure(p, r)
    print(f"  Precision = {p:.4f}")
    print(f"  Recall    = {r:.4f}")
    print(f"  F1-score  = {f1:.4f}")
    print()

    for k in [1, 2, 3, 5]:
        pk = EvaluationMetrics.precision_at_k(retrieved, relevant, k)
        print(f"  P@{k} = {pk:.4f}")
    print()

    ap = EvaluationMetrics.average_precision(retrieved, relevant)
    rr = EvaluationMetrics.reciprocal_rank(retrieved, relevant)
    rp = EvaluationMetrics.r_precision(retrieved, relevant)
    print(f"  Average Precision (AP) = {ap:.4f}")
    print(f"  Reciprocal Rank (RR)   = {rr:.4f}")
    print(f"  R-Precision            = {rp:.4f}")
    print()

    # NDCG 示例 (多级相关度)
    # 假设相关度: doc3=2(高度相关), doc7=2, doc5=1(部分相关), 其他=0
    rankings = [2, 1, 2, 0, 0]  # 对应 retrieved 列表中各文档的相关度
    for k in [3, 5]:
        ndcg = EvaluationMetrics.ndcg_at_k(rankings, k)
        dcg = EvaluationMetrics.dcg_at_k(rankings, k)
        print(f"  DCG@{k}  = {dcg:.4f}")
        print(f"  NDCG@{k} = {ndcg:.4f}")
    print()

    # MAP 示例 (多个查询)
    queries_results = [
        ([3, 5, 7, 6, 10], {3, 7}),       # 查询1
        ([5, 6, 3, 10, 1], {5, 6}),        # 查询2
        ([9, 3, 7, 2, 4], {9, 7, 3}),      # 查询3
    ]
    map_score = EvaluationMetrics.mean_average_precision(queries_results)
    print(f"  MAP (3个查询) = {map_score:.4f}")

    # 11点插值P-R曲线
    pr_curve = EvaluationMetrics.interpolated_precision_recall(retrieved, relevant)
    print(f"\n  11点插值 Precision-Recall 曲线:")
    for recall_level, prec in pr_curve.items():
        bar = "█" * int(prec * 20)
        print(f"    R={recall_level:.1f}: P={prec:.4f} {bar}")
    print()

    # ================================================================
    # 测试2: BM25 检索
    # ================================================================
    print("=" * 70)
    print("  测试2: BM25 概率检索 (Okapi BM25)")
    print("  【ch9&11: BM25 评分 vs TF-IDF 对比】")
    print("=" * 70)

    bm25 = BM25Scorer(engine.index, k1=1.5, b=0.75)
    query = "information retrieval search"
    print(f"\n  查询: '{query}'")
    print(f"  BM25 参数: k1={bm25.k1}, b={bm25.b}")
    print(f"  平均文档长度: {bm25.avgdl:.1f} tokens")

    # BM25 IDF 对比
    print(f"\n  BM25 IDF vs 标准 IDF:")
    preprocessor = TextPreprocessor()
    for t in ["information", "retrieval", "search", "the"]:
        df = engine.index.get_df(t)
        bm25_idf = bm25.idf_bm25(t)
        std_idf = TFIDFScorer.idf(engine.index.doc_count, df) if df > 0 else 0
        print(f"    '{t}': DF={df}, BM25_IDF={bm25_idf:.4f}, 标准IDF={std_idf:.4f}")

    results = bm25.search(query, k=5)
    print(f"\n  BM25 Top-5 结果:")
    for score, doc_id in results:
        print(f"    Doc {doc_id} [BM25={score:.4f}]: \"{documents[doc_id][:60]}...\"")
    print()

    # ================================================================
    # 测试3: 语言模型检索
    # ================================================================
    print("=" * 70)
    print("  测试3: 语言模型检索 (Language Model)")
    print("  【ch11: Dirichlet 平滑 vs Jelinek-Mercer 平滑】")
    print("=" * 70)

    lm_dir = LanguageModelScorer(engine.index, method="dirichlet", mu=100)
    lm_jm = LanguageModelScorer(engine.index, method="jelinek-mercer", lambda_param=0.7)

    query = "search engine retrieval"
    print(f"\n  查询: '{query}'")

    print(f"\n  Dirichlet 平滑 (μ=100) Top-5:")
    results_dir = lm_dir.search(query, k=5)
    for score, doc_id in results_dir:
        print(f"    Doc {doc_id} [logP={score:.4f}]: \"{documents[doc_id][:60]}...\"")

    print(f"\n  Jelinek-Mercer 平滑 (λ=0.7) Top-5:")
    results_jm = lm_jm.search(query, k=5)
    for score, doc_id in results_jm:
        print(f"    Doc {doc_id} [logP={score:.4f}]: \"{documents[doc_id][:60]}...\"")
    print()

    # ================================================================
    # 测试4: 文本分类
    # ================================================================
    print("=" * 70)
    print("  测试4: 文本分类 (Naive Bayes / kNN / Rocchio)")
    print("  【ch13&14: 多项式朴素贝叶斯, kNN, 质心分类】")
    print("=" * 70)

    # 训练数据: 科技 vs 体育
    train_docs = [
        "machine learning algorithms train neural network deep learning",
        "information retrieval search engine index documents query",
        "natural language processing text mining classification",
        "computer science data structure algorithm programming",
        "artificial intelligence robot autonomous system",
        "football soccer goal score championship league match",
        "basketball player team game court dribble shoot",
        "tennis racket serve volley match tournament champion",
        "swimming pool race medal olympic athlete record",
        "baseball pitcher home run bat innings strike",
    ]
    train_labels = ["tech", "tech", "tech", "tech", "tech",
                    "sports", "sports", "sports", "sports", "sports"]

    test_docs = [
        "search algorithm optimization information system",
        "player score goal match championship",
        "deep learning model training data neural network",
    ]

    # Naive Bayes
    nb = NaiveBayesClassifier()
    nb.train(train_docs, train_labels)
    print(f"\n  --- Naive Bayes 分类 ---")
    for doc in test_docs:
        pred, scores = nb.predict(doc)
        print(f"  \"{doc[:50]}...\"")
        print(f"    → 预测: {pred} (tech={scores['tech']:.2f}, sports={scores['sports']:.2f})")

    # kNN
    knn = KNNClassifier(k=3)
    knn.train(train_docs, train_labels)
    print(f"\n  --- kNN 分类 (k=3) ---")
    for doc in test_docs:
        pred, neighbors = knn.predict(doc)
        neighbor_str = ", ".join(f"{label}({sim:.3f})" for sim, label in neighbors)
        print(f"  \"{doc[:50]}...\"")
        print(f"    → 预测: {pred} | 邻居: [{neighbor_str}]")

    # Rocchio
    rocchio = RocchioClassifier()
    rocchio.train(train_docs, train_labels)
    print(f"\n  --- Rocchio 质心分类 ---")
    for doc in test_docs:
        pred, scores = rocchio.predict(doc)
        print(f"  \"{doc[:50]}...\"")
        print(f"    → 预测: {pred} (tech={scores['tech']:.4f}, sports={scores['sports']:.4f})")
    print()

    # ================================================================
    # 测试5: 学习排序
    # ================================================================
    print("=" * 70)
    print("  测试5: 学习排序 (Learning to Rank)")
    print("  【ch15: 决策树 + Pointwise LTR】")
    print("=" * 70)

    ltr = LearningToRank(engine.index)

    # 构造训练数据: (查询, 文档ID, 相关度)
    training_data = [
        ("information retrieval", 3, 3.0),
        ("information retrieval", 7, 2.5),
        ("information retrieval", 5, 1.5),
        ("information retrieval", 6, 1.0),
        ("information retrieval", 1, 0.0),
        ("search engine", 5, 3.0),
        ("search engine", 6, 2.5),
        ("search engine", 3, 1.0),
        ("search engine", 9, 0.5),
        ("search engine", 1, 0.0),
        ("ecology spiders", 4, 3.0),
        ("ecology spiders", 8, 2.5),
        ("ecology spiders", 1, 0.0),
        ("ecology spiders", 2, 0.0),
    ]

    ltr.train(training_data)

    # 测试: 对新查询排序
    test_query = "information retrieval search"
    candidates = list(documents.keys())
    print(f"\n  查询: '{test_query}'")
    print(f"  特征: {ltr.ranker.feature_names}")

    results = ltr.rank(test_query, candidates, k=5)
    print(f"\n  LTR Top-5 排序结果:")
    for score, doc_id in results:
        features = ltr.extract_features(test_query, doc_id)
        print(f"    Doc {doc_id} [LTR={score:.4f}]: \"{documents[doc_id][:50]}...\"")
        print(f"      特征: BM25={features[0]:.3f}, TFIDF={features[1]:.3f}, "
              f"覆盖率={features[2]:.2f}, 长度={features[3]:.2f}")
    print()

    # ================================================================
    # 测试6: PageRank
    # ================================================================
    print("=" * 70)
    print("  测试6: 链接分析 (PageRank & HITS)")
    print("  【ch21: Random Surfer Model, Hubs & Authorities】")
    print("=" * 70)

    # 示例网页图
    web_graph = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": ["A"],
        "D": ["C"],
        "E": ["C", "D"],
        "F": ["B", "E"],
    }

    print(f"\n  网页链接图:")
    for node, links in web_graph.items():
        print(f"    {node} → {links}")

    pr = PageRank(damping=0.85, max_iter=100)
    scores, iterations, history = pr.compute(web_graph)
    print(f"\n  PageRank 结果 (d=0.85, 迭代 {iterations} 次收敛):")
    sorted_pr = sorted(scores.items(), key=lambda x: -x[1])
    for node, score in sorted_pr:
        bar = "█" * int(score * 50)
        print(f"    {node}: {score:.6f} {bar}")

    # HITS
    print(f"\n  HITS (Hubs & Authorities) 结果:")
    hits = HITS()
    hub_scores, auth_scores, hits_iter = hits.compute(web_graph)
    print(f"  (迭代 {hits_iter} 次收敛)")
    print(f"\n  Authority 排名 (好的信息源):")
    sorted_auth = sorted(auth_scores.items(), key=lambda x: -x[1])
    for node, score in sorted_auth:
        print(f"    {node}: auth={score:.6f}")
    print(f"\n  Hub 排名 (好的导航页):")
    sorted_hub = sorted(hub_scores.items(), key=lambda x: -x[1])
    for node, score in sorted_hub:
        print(f"    {node}: hub={score:.6f}")
    print()

    # ================================================================
    # 测试7: 个性化检索
    # ================================================================
    print("=" * 70)
    print("  测试7: 个性化检索 (Search Personalization)")
    print("  【personalization: 用户画像 + 重排序】")
    print("=" * 70)

    personalizer = SearchPersonalizer(engine.index, alpha=0.6)

    # 模拟两个用户
    # 用户A: 对 NLP/IR 感兴趣（点击过文档 3, 7）
    # 用户B: 对生态学感兴趣（点击过文档 4, 8）
    personalizer.update_profile("user_A", [3, 7])
    personalizer.update_profile("user_B", [4, 8])

    query = "web search"
    original_results = engine.ranked_search(query, k=5)
    print(f"\n  查询: '{query}'")
    print(f"  原始排序 (未个性化):")
    for score, doc_id in original_results:
        print(f"    Doc {doc_id} [score={score:.4f}]: \"{documents[doc_id][:60]}\"")

    print(f"\n  用户A (兴趣: IR/NLP) 个性化后:")
    reranked_a = personalizer.personalize("user_A", original_results)
    for final, doc_id, orig, pers in reranked_a:
        print(f"    Doc {doc_id} [final={final:.4f}, orig={orig:.4f}, personal={pers:.4f}]")

    print(f"\n  用户B (兴趣: 生态学) 个性化后:")
    reranked_b = personalizer.personalize("user_B", original_results)
    for final, doc_id, orig, pers in reranked_b:
        print(f"    Doc {doc_id} [final={final:.4f}, orig={orig:.4f}, personal={pers:.4f}]")

    print(f"\n  → 同一查询，不同用户看到不同的排序！(α={personalizer.alpha})")
    print()

    print("=" * 70)
    print("  所有后半期测试完成！(All Part 2 tests completed)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
