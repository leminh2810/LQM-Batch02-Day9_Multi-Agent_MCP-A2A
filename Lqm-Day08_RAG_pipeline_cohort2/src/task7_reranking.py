import os
import numpy as np
from typing import Optional, List, Dict


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Hàm phụ trợ tính độ tương đồng Cosine giữa hai vector."""
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina Reranker API (Yêu cầu JINA_API_KEY).
    Nếu không có API key, hệ thống tự động giữ nguyên thứ hạng ban đầu làm phương án dự phòng.
    """
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        print("⚠ Không tìm thấy JINA_API_KEY trong Environment. Trả về thứ hạng retrieval gốc.")
        return candidates[:top_k]

    import requests
    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {jina_api_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k
            }
        )
        if response.status_code == 200:
            reranked = response.json()["results"]
            return [
                {**candidates[r["index"]], "score": float(r["relevance_score"])}
                for r in reranked
            ]
        else:
            print(f"⚠ API Jina lỗi ({response.status_code}): {response.text}")
            return candidates[:top_k]
    except Exception as e:
        print(f"⚠ Lỗi kết nối Jina Reranker: {str(e)}")
        return candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance (MMR) — Chọn kết quả vừa liên quan vừa đa dạng.
    Công thức: MMR = argmax [ λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_doc)) ]
    
    Yêu cầu các dict trong candidates phải chứa trường 'embedding'.
    """
    if not candidates:
        return []

    # Đảm bảo tất cả các ứng viên đều có vector embedding để tính toán
    for c in candidates:
        if "embedding" not in c:
            raise ValueError("Mỗi candidate cần phải có trường 'embedding' để chạy thuật toán MMR.")

    selected_indices: List[int] = []
    remaining_indices = list(range(len(candidates)))

    # Vòng lặp chọn ra top_k tài liệu thỏa mãn MMR
    for _ in range(min(top_k, len(candidates))):
        best_score = float('-inf')
        best_idx = -1

        for idx in remaining_indices:
            # 1. Tính toán độ liên quan (Relevance) với Query
            relevance = cosine_similarity(query_embedding, candidates[idx]["embedding"])

            # 2. Tính độ trùng lặp lớn nhất với các tài liệu ĐÃ CHỌN (Diversity)
            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sim = cosine_similarity(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                if sim > max_sim_to_selected:
                    max_sim_to_selected = sim

            # 3. Tính điểm MMR phối hợp
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx != -1:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            # Cập nhật điểm score mới chính là điểm MMR phục vụ cho việc hiển thị
            candidates[best_idx]["score"] = round(best_score, 4)

    return [candidates[i] for i in selected_indices]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) — Gộp kết quả từ nhiều bộ tìm kiếm (Semantic & Lexical).
    Công thức: RRF(d) = Σ (1 / (k + rank_r(d)))
    """
    rrf_scores = {}    # Key: content string -> Value: RRF score
    metadata_map = {}  # Key: content string -> Value: full doc dict

    # Duyệt qua từng danh sách xếp hạng (ví dụ: list từ BM25, list từ Semantic)
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            content_key = item["content"]
            
            # Tính điểm tích lũy nghịch đảo thứ hạng
            current_score = rrf_scores.get(content_key, 0.0)
            rrf_scores[content_key] = current_score + (1.0 / (k + rank))
            
            # Lưu lại thông tin metadata gốc
            metadata_map[content_key] = item

    # Sắp xếp các tài liệu giảm dần theo điểm RRF thu được
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_docs[:top_k]:
        final_item = metadata_map[content].copy()
        final_item["score"] = round(score, 5)  # Gán điểm score tổng hợp RRF mới
        results.append(final_item)

    return results


# =============================================================================
# Unified Main Interface (Hàm gọi chính)
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
    query_embedding: Optional[list[float]] = None,
    ranked_lists: Optional[list[list[dict]]] = None,
) -> list[dict]:
    """
    Hàm Wrapper hợp nhất các phương pháp Rerank cho Pipeline RAG.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    
    elif method == "mmr":
        if query_embedding is None:
            raise ValueError("Vui lòng truyền 'query_embedding' (list các float) khi chọn phương pháp MMR.")
        return rerank_mmr(query_embedding, candidates, top_k)
    
    elif method == "rrf":
        if ranked_lists is None:
            raise ValueError("Vui lòng truyền danh sách các bảng kết quả 'ranked_lists' khi chọn phương pháp RRF.")
        return rerank_rrf(ranked_lists, top_k)
    
    else:
        raise ValueError(f"Không nhận diện được phương pháp rerank: {method}")


if __name__ == "__main__":
    print("=== MODULE RERANKING (TESTING MODE) ===")

    # Giả lập dữ liệu text cho cuộc thử nghiệm
    doc_a = "Công an bắt khẩn cấp ca sĩ Chi Dân vì hành vi tàng trữ và tổ chức sử dụng ma túy đá tại chung cư."
    doc_b = "Nghệ sĩ vướng vòng lao lý và đánh mất sự nghiệp âm nhạc đỉnh cao vì các vết trượt ma túy."
    doc_c = "Chi tiết hình phạt và khung hình sự quy định tại Điều 248 Bộ luật hình sự về tội tàng trữ ma túy."

    # Giả lập các vector embedding 3 chiều đại diện
    emb_query = [1.0, 1.0, 0.0]
    emb_a     = [0.9, 0.9, 0.1]
    emb_b     = [0.5, 0.4, 0.2]
    emb_c     = [0.8, 0.8, 0.1]  # Khá tương đồng với doc_a về vector, dễ gây trùng lặp nội dung

    # 1. Thử nghiệm thuật toán MMR (Chống trùng lặp thông tin)
    print("\n[1] Thử nghiệm thuật toán MMR (Duy trì tính Đa dạng dữ liệu):")
    candidates_for_mmr = [
        {"content": doc_a, "score": 0.9, "embedding": emb_a, "metadata": {"source": "file_1"}},
        {"content": doc_c, "score": 0.85, "embedding": emb_c, "metadata": {"source": "file_3"}}, # Trùng lặp ý với doc_a
        {"content": doc_b, "score": 0.6, "embedding": emb_b, "metadata": {"source": "file_2"}}, # Nội dung khác biệt, đa dạng
    ]
    
    mmr_results = rerank(
        query="ca sĩ Chi Dân tàng trữ ma túy",
        candidates=candidates_for_mmr,
        top_k=2,
        method="mmr",
        query_embedding=emb_query
    )
    for idx, r in enumerate(mmr_results, 1):
        print(f"   {idx}. Score: {r['score']} -> {r['content'][:60]}...")

    # 2. Thử nghiệm thuật toán RRF (Gộp kết quả từ BM25 và Semantic)
    print("\n[2] Thử nghiệm thuật toán RRF (Hợp nhất kết quả Lexical + Semantic):")
    
    # Kết quả giả định xếp hạng từ module BM25 (Ưu tiên từ khóa chính xác)
    bm25_list = [
        {"content": doc_c, "score": 25.4, "metadata": {}},
        {"content": doc_a, "score": 18.2, "metadata": {}},
    ]
    # Kết quả giả định xếp hạng từ module Semantic (Ưu tiên ngữ nghĩa cốt truyện)
    semantic_list = [
        {"content": doc_a, "score": 0.85, "metadata": {}},
        {"content": doc_b, "score": 0.72, "metadata": {}},
    ]

    rrf_results = rerank(
        query="hình phạt tàng trữ ma túy",
        candidates=[],  # RRF lấy từ ranked_lists
        top_k=3,
        method="rrf",
        ranked_lists=[bm25_list, semantic_list]
    )
    for idx, r in enumerate(rrf_results, 1):
        print(f"   {idx}. Điểm RRF: {r['score']} -> {r['content'][:60]}...")