"""
Mini Search Engine - Web 可视化界面
====================================
基于 Flask 的 Web 前端，对 mini_search_engine.py 中的核心模块进行可视化展示。
"""

from flask import Flask, render_template, request, jsonify
from mini_search_engine import (
    MiniSearchEngine, TextPreprocessor, TFIDFScorer, SpellingCorrector
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


@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


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


if __name__ == "__main__":
    app.run(debug=True, port=5001)
