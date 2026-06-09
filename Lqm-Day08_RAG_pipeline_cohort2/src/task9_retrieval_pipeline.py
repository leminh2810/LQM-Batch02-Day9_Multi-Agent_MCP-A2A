"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    print(f"🚀 Bắt đầu chạy Hybrid Pipeline cho câu truy vấn: '{query}'")
    
    # Step 1: Song song lấy dữ liệu từ Semantic (Dense) và Lexical (Sparse)
    # Lấy top_k * 2 để đảm bảo sau khi merge và rerank không bị thiếu hụt dữ liệu chất lượng
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)
    
    # Khởi tạo danh sách kết quả sau cùng
    final_results = []
    
    # Step 2: Merge bằng thuật toán Reciprocal Rank Fusion (RRF) từ Task 7
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    
    # Gán nhãn nguồn mặc định là hybrid trước khi đánh giá chất lượng
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Tiến hành Rerank để tái phân phối điểm số chính xác bằng Cross-Encoder
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Step 4: Kiểm tra điều kiện chất lượng dựa trên điểm số (Score Threshold) -> Fallback sang PageIndex
    best_score = final_results[0]["score"] if final_results else 0.0
    
    if not final_results or best_score < score_threshold:
        print(f"  ⚠ Điểm số Hybrid RAG quá thấp ({best_score:.3f}) hoặc rỗng, "
              f"dưới ngưỡng quy định ({score_threshold}).")
        print("  🔄 Kích hoạt chế độ Fallback: Chuyển hướng sang cấu trúc cây PageIndex Vectorless...")
        
        # Gọi module PageIndex xử lý duyệt cây/hội thoại tài liệu thô
        fallback_results = pageindex_search(query, top_k=top_k)
        
        # Đảm bảo nguồn được ghi nhận rõ ràng là 'pageindex' phục vụ đánh giá sau này
        for item in fallback_results:
            item["source"] = "pageindex"
            
        if fallback_results:
            return fallback_results[:top_k]
        else:
            print("  ✕ PageIndex không tìm thấy dữ liệu bổ sung phù hợp.")
            return final_results[:top_k]

    print(f"  ✓ Tìm thấy kết quả Hybrid tối ưu vượt ngưỡng chất lượng ({best_score:.3f} >= {score_threshold}).")
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print("\n" + "="*70)
        print(f"Query: {q}")
        print("="*70)
        results = retrieve(q, top_k=3)
        print(f"\n📈 KẾT QUẢ TRẢ VỀ ĐẦU RA (Top 3):")
        for i, r in enumerate(results, 1):
            score_val = r.get('score', 0.0)
            source_val = r.get('source', 'unknown')
            content_snippet = r.get('content', '').replace('\n', ' ').strip()
            print(f"  {i}. [{score_val:.3f}] [{source_val}] {content_snippet[:100]}...")