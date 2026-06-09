"""
Task 5 — Semantic Search Module (Dense Retrieval Pipeline).

Nhiệm vụ:
    1. Mã hóa câu hỏi truy vấn (Query Embedding) bằng mô hình ngôn ngữ dạng ONNX.
    2. Thực hiện tìm kiếm không gian vector trên ChromaDB.
    3. Chuẩn hóa khoảng cách hình học (Distance) về thang điểm tương đồng [0.0 - 1.0].
"""

import os
import warnings
from pathlib import Path

# Đặt bộ lọc ignore ở ĐẦU FILE để bắt được cảnh báo từ các thư viện khi import
warnings.filterwarnings("ignore", category=UserWarning, module="fastembed")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Bây giờ mới tiến hành import các thư viện nặng
from fastembed import TextEmbedding
from langchain_chroma import Chroma

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

# ... (Các phần code Class CustomFastEmbed và hàm semantic_search giữ nguyên như cũ)

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

# Định nghĩa đường dẫn tới Vector Store đã index ở các task trước
CHROMA_DB_DIR = Path(__file__).parent.parent / "data" / "vectorstore" / "chroma_db"


class CustomFastEmbed:
    """
    Adapter Class bọc thư viện fastembed gốc (chạy thuần ONNX, không PyTorch)
    Được đồng bộ cấu trúc chính xác để mã hóa câu hỏi truy vấn.
    """
    def __init__(self):
        model_candidates = [
            "BAAI/bge-m3",
            "bge-m3",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "nomic-ai/nomic-embed-text-v1.5",
            "BAAI/bge-small-en-v1.5"
        ]
        
        self.model = None
        for model_name in model_candidates:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*now uses mean pooling instead of CLS embedding.*",
                        category=UserWarning,
                    )
                    self.model = TextEmbedding(model_name=model_name)
                break
            except (ValueError, Exception):
                continue
                
        if self.model is None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*now uses mean pooling instead of CLS embedding.*",
                    category=UserWarning,
                )
                self.model = TextEmbedding()
        
    def embed_documents(self, texts):
        return list(self.model.embed(texts))
        
    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Thực hiện semantic search (dense retrieval) trên ChromaDB vector store.
    Tự động chuẩn hóa khoảng cách L2/Cosine của ChromaDB về thang điểm 1 chuẩn [0.0 - 1.0].
    """
    if not CHROMA_DB_DIR.exists():
        print(f"❌ Không tìm thấy cơ sở dữ liệu VectorStore tại: {CHROMA_DB_DIR}.")
        return []

    # 1. Khởi tạo bộ embedding
    embeddings = CustomFastEmbed()

    # 2. Kết nối tới ChromaDB
    vector_store = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings
    )

    # 3. Sử dụng hàm gốc `similarity_search_with_score` để lấy trực tiếp khoảng cách (Distance) thô
    raw_results = vector_store.similarity_search_with_score(query, k=top_k)

    formatted_results = []
    for doc, raw_score in raw_results:
        distance = abs(float(raw_score))
        
        # Công thức chuẩn hóa khoảng cách L2 thô về Thang điểm tương đồng từ 0.0 đến 1.0
        relevance_score = 1.0 / (1.0 + (distance / 10.0)) 
        
        # Tạo cận dưới an toàn để điểm số không bị quá thấp khi văn bản vẫn có từ khóa trùng khớp
        if relevance_score < 0.5 and distance < 10.0:
            relevance_score = 0.5 + (0.5 * (1.0 - (distance / 10.0)))

        # Hỗ trợ cấu trúc trường kép "content" và "text" để đồng bộ tuyệt đối với Task 4 và bộ sinh câu hỏi
        formatted_results.append({
            "content": doc.page_content,
            "text": doc.page_content,
            "score": round(relevance_score, 4),  # Làm tròn 4 chữ số thập phân
            "metadata": doc.metadata
        })

    # 4. Sắp xếp danh sách giảm dần (Descending) theo score
    formatted_results.sort(key=lambda x: x["score"], reverse=True)

    return formatted_results


if __name__ == "__main__":
    print("=== MODULE SEMANTIC SEARCH (TESTING MODE) ===")
    
    test_query = "Cơ quan công an đã tạm giữ những nghệ sĩ nào liên quan đến ma túy đá?"
    k_elements = 3
    
    print(f"Câu hỏi truy vấn: '{test_query}'")
    print(f"Đang tìm kiếm top {k_elements} đoạn văn bản phù hợp nhất...\n")
    
    results = semantic_search(query=test_query, top_k=k_elements)
    
    if not results:
        print("Không tìm thấy kết quả nào hoặc Database trống.")
    else:
        for idx, item in enumerate(results, 1):
            print(f"--- [KẾT QUẢ {idx}] ---")
            print(f"📍 Điểm tương đồng (Thang điểm 1): {item['score']}")
            # Khắc phục lỗi hiển thị nguồn bằng cách đọc cả 2 khả năng đặt tên Key của Task 4
            source_file = item['metadata'].get('filename') or item['metadata'].get('source') or 'Unknown'
            print(f"📂 Nguồn file: {source_file}")
            print(f"📝 Nội dung chunk:\n{item['content']}")
            print("-" * 50)
