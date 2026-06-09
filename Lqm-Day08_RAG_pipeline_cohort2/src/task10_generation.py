"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "Tôi không thể xác minh thông tin này từ nguồn hiện có"
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Sử dụng Absolute Import để chạy độc lập hoặc chạy module một cách linh hoạt
try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn tham số
# =============================================================================

# top_k: Số lượng văn bản (chunks) được đưa vào làm ngữ cảnh (context) cho LLM.
# Chọn 5 vì: Cung cấp đầy đủ bằng chứng dữ liệu từ cả 2 nguồn hybrid mà không vượt
# quá giới hạn cửa sổ ngữ cảnh, tránh làm loãng thông tin gây nhiễu cho mô hình.
TOP_K = 5

# top_p (nucleus sampling): Tổng xác suất tích lũy của các từ kế tiếp được cân nhắc.
# Chọn 0.9 vì: Cho phép mô hình linh hoạt lựa chọn từ ngữ tự nhiên, mạch lạc nhưng
# vẫn giữ được tính khuôn mẫu và chính xác cao cần thiết cho văn bản pháp lý.
TOP_P = 0.9

# temperature: Định hình độ sáng tạo, ngẫu nhiên của nội dung đầu ra.
# Chọn 0.3 vì: Hệ thống RAG (nhất là mảng pháp luật) đòi hỏi tính xác thực (factual),
# câu trả lời cần bám sát nghiêm ngặt dữ liệu đầu vào và hạn chế tối đa ảo tưởng (hallucination).
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [13_2026_NQ-HDND_703498.pdf, Page 2]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context. Do not use external knowledge.
- Every factual claim MUST have a citation to the specific document index or source name provided.
- If context is insufficient or has no relevant data, say 'Tôi không thể xác minh thông tin này từ nguồn hiện có' clearly.
- Structure your answer with clear paragraphs."""


# =============================================================================
# DOCUMENT REORDERING (Tránh hiện tượng Lost in the Middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: Đặt chunks quan trọng nhất (score cao) ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (Điểm cao nhất lên đầu, điểm cao nhì xuống cuối, điểm thấp nhất rơi vào giữa)

    Args:
        chunks: Danh sách các chunk đã được sắp xếp giảm dần theo Score từ bộ Rerank.

    Returns:
        Danh sách các chunk đã được đổi chỗ để tối ưu hóa sự chú ý (Attention) của LLM.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = [None] * len(chunks)
    left = 0
    right = len(chunks) - 1

    # Phân bổ xen kẽ: Phần tử tốt nhất sang trái (đầu), tốt nhì sang phải (cuối)
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            reordered[left] = chunk
            left += 1
        else:
            reordered[right] = chunk
            right -= 1

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành một chuỗi context hoàn chỉnh cho prompt.
    Mỗi đoạn văn bản đều được đóng nhãn (Source Labels) rõ ràng để LLM đối chiếu trích dẫn.

    Args:
        chunks: Danh sách các dictionary chứa văn bản và metadata.

    Returns:
        Chuỗi ngữ cảnh đã format.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Trích xuất linh hoạt thông tin nguồn từ metadata của Vector DB hoặc PageIndex
        metadata = chunk.get("metadata", {})
        source = metadata.get("filename") or metadata.get("source") or f"Document_{i}"
        
        # Bổ sung thông tin trang hoặc tiêu đề mục nếu có để tăng độ chi tiết citation
        page_info = f" | Page: {metadata.get('page_index')}" if metadata.get('page_index') is not None else ""
        node_info = f" | Section: {metadata.get('node_title')}" if metadata.get('node_title') else ""
        
        context_parts.append(
            f"[Document Index {i} | Source: {source}{page_info}{node_info}]\n"
            f"Content: {chunk['content'].strip()}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION WITH CITATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Gọi Retrieval Pipeline (Task 9) song song + rerank để lấy chunks chất lượng.
        2. Tái cấu trúc vị trí chunks (Reorder) nhằm tối ưu hóa Attention.
        3. Định dạng context kèm nhãn tài liệu chi tiết.
        4. Xây dựng cấu trúc prompt hoàn chỉnh gửi tới LLM.
        5. Gọi OpenAI API (hoặc các dịch vụ LLM tương thích) với tham số factual.
        6. Kiểm tra tính xác thực và trả về câu trả lời.

    Args:
        query: Câu hỏi của người dùng.
        top_k: Số lượng ngữ cảnh tối đa cần trích xuất.

    Returns:
        Dictionary chứa câu trả lời, danh sách nguồn tham chiếu và phương thức tìm kiếm.
    """
    # Step 1: Lấy các đoạn văn bản liên quan nhất từ Pipeline tích hợp của Task 9
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có",
            "sources": [],
            "retrieval_source": "none"
        }

    # Trích xuất nhãn phương thức tìm kiếm được ghi nhận từ hệ thống
    retrieval_source = chunks[0].get("source", "hybrid")

    # Step 2: Áp dụng thuật toán Reorder chống hiện tượng mất tập trung ở giữa prompt
    reordered_chunks = reorder_for_llm(chunks)

    # Step 3: Đóng gói và gắn thẻ định danh nguồn cho khối văn bản ngữ cảnh
    context_str = format_context(reordered_chunks)

    # Step 4: Tạo thông điệp đầu vào tích hợp ngữ cảnh
    user_message = (
        f"Dưới đây là các tài liệu ngữ cảnh được trích xuất từ hệ thống dữ liệu đáng tin cậy:\n"
        f"=================== NGỮ CẢNH TÀI LIỆU ===================\n"
        f"{context_str}\n"
        f"=========================================================\n\n"
        f"Dựa hoàn toàn vào ngữ cảnh trên, hãy trả lời câu hỏi sau:\n"
        f"Question: {query}"
    )

    # Step 5: Kết nối và gọi API của OpenAI để sinh văn bản có kiểm soát hành vi
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return {
            "answer": "❌ Lỗi cấu hình hệ thống: Thiếu OPENAI_API_KEY trong file .env",
            "sources": chunks,
            "retrieval_source": retrieval_source
        }

    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Có thể tùy biến thành gpt-4o hoặc các model phù hợp nhu cầu của bạn
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"❌ Lỗi kết nối hệ thống xử lý ngôn ngữ lớn (LLM): {str(e)}"

    # Step 6: Đóng gói dữ liệu đầu ra hoàn chỉnh cho Pipeline RAG sản xuất
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source
    }


if __name__ == "__main__":
    # Danh sách các câu hỏi kiểm định thực tế theo kịch bản bài toán
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    print("=== THỬ NGHIỆM ĐẦU CUỐI: PIPELINE RAG CÓ CITATION ===")
    
    for q in test_queries:
        print(f"\n{'-'*80}")
        print(f"👉 CÂU HỎI USER: {q}")
        print(f"{'-'*80}")
        
        result = generate_with_citation(q)
        
        print(f"\n🤖 CÂU TRẢ LỜI CỦA MÔ HÌNH (LLM):")
        print(result['answer'])
        print(f"\n📊 [Thông tin bổ trợ: Thu nhận {len(result['sources'])} chunks văn bản gốc | Thông qua luồng: {result['retrieval_source']}]")