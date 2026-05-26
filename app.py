"""
Mini Search Engine - Web 可视化界面
====================================
基于 Flask 的 Web 前端，对 mini_search_engine.py 中的核心模块进行可视化展示。
"""

from flask import Flask, render_template, request, jsonify
from mini_search_engine import (
    MiniSearchEngine, TextPreprocessor, TFIDFScorer, SpellingCorrector
)
from mini_search_engine_v2 import (
    EvaluationMetrics, BM25Scorer, LanguageModelScorer,
    NaiveBayesClassifier, KNNClassifier, RocchioClassifier,
    LearningToRank, PageRank, HITS, SearchPersonalizer
)
from collections import defaultdict
import math

app = Flask(__name__)

# ---- 初始化搜索引擎与语料 ----
DOCUMENTS = {
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

engine = MiniSearchEngine()
engine.build(DOCUMENTS)

# ---- 初始化后半期模块 ----
bm25_scorer = BM25Scorer(engine.index, k1=1.5, b=0.75)
lm_dirichlet = LanguageModelScorer(engine.index, method="dirichlet", mu=100)
lm_jm = LanguageModelScorer(engine.index, method="jelinek-mercer", lambda_param=0.7)
personalizer = SearchPersonalizer(engine.index, alpha=0.6)

# 预设用户画像
personalizer.update_profile("user_A", [3, 7])  # IR/NLP 兴趣
personalizer.update_profile("user_B", [4, 8])  # 生态学兴趣

# 训练文本分类器
TRAIN_DOCS = [
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
TRAIN_LABELS = ["tech", "tech", "tech", "tech", "tech",
                "sports", "sports", "sports", "sports", "sports"]

nb_classifier = NaiveBayesClassifier()
nb_classifier.train(TRAIN_DOCS, TRAIN_LABELS)
knn_classifier = KNNClassifier(k=3)
knn_classifier.train(TRAIN_DOCS, TRAIN_LABELS)
rocchio_classifier = RocchioClassifier()
rocchio_classifier.train(TRAIN_DOCS, TRAIN_LABELS)

# 训练 LTR
ltr = LearningToRank(engine.index)
ltr_training = [
    ("information retrieval", 3, 3.0), ("information retrieval", 7, 2.5),
    ("information retrieval", 5, 1.5), ("information retrieval", 6, 1.0),
    ("information retrieval", 1, 0.0),
    ("search engine", 5, 3.0), ("search engine", 6, 2.5),
    ("search engine", 3, 1.0), ("search engine", 9, 0.5), ("search engine", 1, 0.0),
    ("ecology spiders", 4, 3.0), ("ecology spiders", 8, 2.5),
    ("ecology spiders", 1, 0.0), ("ecology spiders", 2, 0.0),
]
ltr.train(ltr_training)


@app.route("/")
def index():
    """主页"""
    return render_template("index_v2.html")


@app.route("/api/index_stats")
def index_stats():
    """返回索引统计信息"""
    vocab = engine.index.vocabulary()
    # 取部分索引样本
    sample = {}
    for term in vocab[:30]:
        postings = engine.index.get_postings(term)
        df = engine.index.get_df(term)
        sample[term] = {
            "df": df,
            "postings": {
                str(doc_id): positions
                for doc_id, positions in sorted(postings.items())
            }
        }
    return jsonify({
        "doc_count": engine.index.doc_count,
        "vocab_size": len(vocab),
        "vocabulary": vocab,
        "documents": {str(k): v for k, v in DOCUMENTS.items()},
        "sample_index": sample,
    })


@app.route("/api/boolean_search")
def boolean_search():
    """布尔 AND 查询，附带 DF 优化过程可视化数据"""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "查询为空"}), 400

    preprocessor = TextPreprocessor()
    tokens = preprocessor.tokenize(q)
    terms = [t[0] for t in tokens]

    # DF 信息
    term_df = []
    for t in terms:
        term_df.append({"term": t, "df": engine.index.get_df(t)})

    # 按 DF 排序后的顺序
    sorted_terms = sorted(term_df, key=lambda x: x["df"])

    # 逐步合并过程
    merge_steps = []
    if sorted_terms:
        current = engine.index.get_doc_ids(sorted_terms[0]["term"])
        merge_steps.append({
            "step": 1,
            "action": f"取 '{sorted_terms[0]['term']}' 的 postings (DF={sorted_terms[0]['df']})",
            "result_ids": current,
            "result_count": len(current),
        })
        for i, item in enumerate(sorted_terms[1:], 2):
            next_list = engine.index.get_doc_ids(item["term"])
            from mini_search_engine import BooleanQueryEngine
            current = BooleanQueryEngine._intersect_two_postings(current, next_list)
            merge_steps.append({
                "step": i,
                "action": f"AND 合并 '{item['term']}' (DF={item['df']})",
                "result_ids": current,
                "result_count": len(current),
            })

    result_docs = []
    for doc_id in (current if sorted_terms else []):
        result_docs.append({"id": doc_id, "text": DOCUMENTS.get(doc_id, "")})

    return jsonify({
        "query": q,
        "terms": terms,
        "term_df": term_df,
        "optimized_order": [x["term"] for x in sorted_terms],
        "merge_steps": merge_steps,
        "results": result_docs,
    })


@app.route("/api/phrase_search")
def phrase_search():
    """短语查询"""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "查询为空"}), 400

    preprocessor = TextPreprocessor()
    tokens = preprocessor.tokenize(q)
    terms = [t[0] for t in tokens]

    # 各词的位置信息（用于可视化）
    term_positions = {}
    for t in terms:
        postings = engine.index.get_postings(t)
        term_positions[t] = {
            str(doc_id): positions
            for doc_id, positions in sorted(postings.items())
        }

    result_ids = engine.phrase_search(q)
    result_docs = []
    for doc_id in result_ids:
        result_docs.append({"id": doc_id, "text": DOCUMENTS.get(doc_id, "")})

    return jsonify({
        "query": q,
        "terms": terms,
        "term_positions": term_positions,
        "results": result_docs,
    })


@app.route("/api/ranked_search")
def ranked_search():
    """排序检索，附带详细打分过程"""
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 5))
    if not q:
        return jsonify({"error": "查询为空"}), 400

    preprocessor = TextPreprocessor()
    tokens = preprocessor.tokenize(q)
    N = engine.index.doc_count

    # 查询词 IDF 信息
    query_tf = defaultdict(int)
    for term, _ in tokens:
        query_tf[term] += 1

    term_info = []
    for term, tf in query_tf.items():
        df = engine.index.get_df(term)
        idf_val = TFIDFScorer.idf(N, df) if df > 0 else 0
        ltc_raw = TFIDFScorer.compute_query_weight(tf, N, df) if df > 0 else 0
        term_info.append({
            "term": term,
            "tf_in_query": tf,
            "df": df,
            "idf": round(idf_val, 4),
            "ltc_raw": round(ltc_raw, 4),
            "in_vocab": df > 0,
        })

    # 执行检索
    results_raw = engine.ranked_search(q, k=k)

    results = []
    max_score = results_raw[0][0] if results_raw else 1
    for score, doc_id in results_raw:
        results.append({
            "id": doc_id,
            "text": DOCUMENTS.get(doc_id, ""),
            "score": round(score, 6),
            "bar_width": round(score / max_score * 100, 1) if max_score > 0 else 0,
        })

    return jsonify({
        "query": q,
        "k": k,
        "N": N,
        "term_info": term_info,
        "results": results,
    })


@app.route("/api/spell_check")
def spell_check():
    """拼写纠错"""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "查询为空"}), 400

    preprocessor = TextPreprocessor()
    tokens = preprocessor.tokenize(q)

    corrections = []
    for term, _ in tokens:
        in_vocab = engine.index.get_df(term) > 0
        suggestion = engine.spell_corrector.correct(term)
        dist = SpellingCorrector.levenshtein_distance(term, suggestion) if suggestion else 0
        corrections.append({
            "original": term,
            "in_vocab": in_vocab,
            "suggestion": suggestion,
            "distance": dist,
        })

    return jsonify({
        "query": q,
        "corrections": corrections,
    })


# ============================================================
# 后半期 API: BM25 检索
# ============================================================

@app.route("/api/bm25_search")
def bm25_search():
    """BM25 排序检索"""
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 5))
    if not q:
        return jsonify({"error": "查询为空"}), 400

    preprocessor = TextPreprocessor()
    tokens = preprocessor.tokenize(q)
    query_terms = [t[0] for t in tokens]
    N = engine.index.doc_count

    # 各词 BM25 IDF
    term_info = []
    for t in query_terms:
        df = engine.index.get_df(t)
        bm25_idf = bm25_scorer.idf_bm25(t)
        std_idf = TFIDFScorer.idf(N, df) if df > 0 else 0
        term_info.append({
            "term": t, "df": df,
            "bm25_idf": round(bm25_idf, 4),
            "std_idf": round(std_idf, 4),
        })

    results_raw = bm25_scorer.search(q, k=k)
    results = []
    max_score = results_raw[0][0] if results_raw else 1
    for score, doc_id in results_raw:
        results.append({
            "id": doc_id, "text": DOCUMENTS.get(doc_id, ""),
            "score": round(score, 4),
            "bar_width": round(score / max_score * 100, 1) if max_score > 0 else 0,
        })

    return jsonify({
        "query": q, "k": k, "N": N,
        "params": {"k1": bm25_scorer.k1, "b": bm25_scorer.b, "avgdl": round(bm25_scorer.avgdl, 2)},
        "term_info": term_info, "results": results,
    })


# ============================================================
# 后半期 API: 语言模型检索
# ============================================================

@app.route("/api/lm_search")
def lm_search():
    """语言模型检索"""
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 5))
    method = request.args.get("method", "dirichlet")
    if not q:
        return jsonify({"error": "查询为空"}), 400

    scorer = lm_dirichlet if method == "dirichlet" else lm_jm
    results_raw = scorer.search(q, k=k)

    results = []
    max_score = results_raw[0][0] if results_raw else -100
    min_score = results_raw[-1][0] if results_raw else -100
    score_range = max_score - min_score if max_score > min_score else 1
    for score, doc_id in results_raw:
        results.append({
            "id": doc_id, "text": DOCUMENTS.get(doc_id, ""),
            "score": round(score, 4),
            "bar_width": round((score - min_score) / score_range * 100, 1) if score_range > 0 else 50,
        })

    params = {"method": method}
    if method == "dirichlet":
        params["mu"] = scorer.mu
    else:
        params["lambda"] = scorer.lambda_param

    return jsonify({"query": q, "k": k, "method": method, "params": params, "results": results})


# ============================================================
# 后半期 API: 检索评价
# ============================================================

@app.route("/api/evaluation")
def evaluation():
    """计算评价指标"""
    q = request.args.get("q", "").strip()
    relevant_str = request.args.get("relevant", "").strip()
    if not q or not relevant_str:
        return jsonify({"error": "需要查询和相关文档ID"}), 400

    relevant = set(int(x) for x in relevant_str.split(",") if x.strip().isdigit())

    # 使用 BM25 检索作为系统输出
    results_raw = bm25_scorer.search(q, k=10)
    retrieved = [doc_id for _, doc_id in results_raw]

    # 构造 graded relevance (相关=2, 不相关=0)
    rankings = [2 if doc_id in relevant else 0 for doc_id in retrieved]

    p = EvaluationMetrics.precision(retrieved, relevant)
    r = EvaluationMetrics.recall(retrieved, relevant)
    f1 = EvaluationMetrics.f_measure(p, r)
    ap = EvaluationMetrics.average_precision(retrieved, relevant)
    rr = EvaluationMetrics.reciprocal_rank(retrieved, relevant)
    rp = EvaluationMetrics.r_precision(retrieved, relevant)

    pk_values = {}
    for k_val in [1, 3, 5, 10]:
        pk_values[f"P@{k_val}"] = round(EvaluationMetrics.precision_at_k(retrieved, relevant, k_val), 4)

    ndcg_values = {}
    for k_val in [3, 5, 10]:
        ndcg_values[f"NDCG@{k_val}"] = round(EvaluationMetrics.ndcg_at_k(rankings, k_val), 4)

    pr_curve = EvaluationMetrics.interpolated_precision_recall(retrieved, relevant)

    retrieved_with_rel = []
    for doc_id in retrieved:
        retrieved_with_rel.append({
            "id": doc_id, "text": DOCUMENTS.get(doc_id, ""),
            "relevant": doc_id in relevant,
        })

    return jsonify({
        "query": q,
        "relevant_set": list(relevant),
        "retrieved": retrieved_with_rel,
        "metrics": {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "AP": round(ap, 4), "RR": round(rr, 4), "R-Precision": round(rp, 4),
        },
        "pk": pk_values,
        "ndcg": ndcg_values,
        "pr_curve": pr_curve,
    })


# ============================================================
# 后半期 API: PageRank & HITS
# ============================================================

@app.route("/api/pagerank")
def pagerank_api():
    """PageRank 计算"""
    damping = float(request.args.get("d", 0.85))
    graph = {
        "A": ["B", "C"], "B": ["C"], "C": ["A"],
        "D": ["C"], "E": ["C", "D"], "F": ["B", "E"],
    }
    pr = PageRank(damping=damping)
    scores, iterations, history = pr.compute(graph)
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    max_s = sorted_scores[0][1] if sorted_scores else 1

    results = [{"node": n, "score": round(s, 6), "bar_width": round(s / max_s * 100, 1)}
               for n, s in sorted_scores]

    return jsonify({
        "graph": graph, "damping": damping,
        "iterations": iterations, "results": results,
    })


@app.route("/api/hits")
def hits_api():
    """HITS 计算"""
    graph = {
        "A": ["B", "C"], "B": ["C"], "C": ["A"],
        "D": ["C"], "E": ["C", "D"], "F": ["B", "E"],
    }
    hits = HITS()
    hub_scores, auth_scores, iterations = hits.compute(graph)

    auth_sorted = sorted(auth_scores.items(), key=lambda x: -x[1])
    hub_sorted = sorted(hub_scores.items(), key=lambda x: -x[1])
    max_auth = auth_sorted[0][1] if auth_sorted else 1
    max_hub = hub_sorted[0][1] if hub_sorted else 1

    return jsonify({
        "graph": graph, "iterations": iterations,
        "authorities": [{"node": n, "score": round(s, 6), "bar_width": round(s / max_auth * 100, 1)} for n, s in auth_sorted],
        "hubs": [{"node": n, "score": round(s, 6), "bar_width": round(s / max_hub * 100, 1)} for n, s in hub_sorted],
    })


# ============================================================
# 后半期 API: 文本分类
# ============================================================

@app.route("/api/classify")
def classify():
    """文本分类"""
    q = request.args.get("q", "").strip()
    method = request.args.get("method", "nb")
    if not q:
        return jsonify({"error": "文本为空"}), 400

    if method == "nb":
        pred, scores = nb_classifier.predict(q)
        return jsonify({"text": q, "method": "Naive Bayes", "prediction": pred,
                        "scores": {k: round(v, 3) for k, v in scores.items()}})
    elif method == "knn":
        pred, neighbors = knn_classifier.predict(q)
        neighbor_list = [{"similarity": round(s, 4), "label": l} for s, l in neighbors]
        return jsonify({"text": q, "method": "kNN (k=3)", "prediction": pred, "neighbors": neighbor_list})
    elif method == "rocchio":
        pred, scores = rocchio_classifier.predict(q)
        return jsonify({"text": q, "method": "Rocchio", "prediction": pred,
                        "scores": {k: round(v, 4) for k, v in scores.items()}})
    return jsonify({"error": "未知方法"}), 400


# ============================================================
# 后半期 API: 学习排序
# ============================================================

@app.route("/api/ltr_search")
def ltr_search():
    """学习排序"""
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 5))
    if not q:
        return jsonify({"error": "查询为空"}), 400

    candidates = list(DOCUMENTS.keys())
    results_raw = ltr.rank(q, candidates, k=k)

    results = []
    max_score = results_raw[0][0] if results_raw else 1
    for score, doc_id in results_raw:
        features = ltr.extract_features(q, doc_id)
        results.append({
            "id": doc_id, "text": DOCUMENTS.get(doc_id, ""),
            "score": round(score, 4),
            "bar_width": round(score / max_score * 100, 1) if max_score > 0 else 0,
            "features": {
                "bm25": round(features[0], 3), "tfidf": round(features[1], 3),
                "coverage": round(features[2], 2), "doc_len": round(features[3], 2),
                "avg_tf": round(features[4], 2), "idf_sum": round(features[5], 3),
            }
        })

    return jsonify({
        "query": q, "k": k,
        "feature_names": ltr.ranker.feature_names,
        "results": results,
    })


# ============================================================
# 后半期 API: 个性化检索
# ============================================================

@app.route("/api/personalized_search")
def personalized_search():
    """个性化检索"""
    q = request.args.get("q", "").strip()
    user = request.args.get("user", "user_A")
    if not q:
        return jsonify({"error": "查询为空"}), 400

    original_results = bm25_scorer.search(q, k=8)
    reranked = personalizer.personalize(user, original_results)

    results = []
    for final, doc_id, orig, pers in reranked:
        results.append({
            "id": doc_id, "text": DOCUMENTS.get(doc_id, ""),
            "final_score": round(final, 4),
            "original_score": round(orig, 4),
            "personal_score": round(pers, 4),
        })

    profile_terms = list(personalizer.user_profiles.get(user, {}).keys())[:15]

    return jsonify({
        "query": q, "user": user, "alpha": personalizer.alpha,
        "profile_top_terms": profile_terms,
        "results": results,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
