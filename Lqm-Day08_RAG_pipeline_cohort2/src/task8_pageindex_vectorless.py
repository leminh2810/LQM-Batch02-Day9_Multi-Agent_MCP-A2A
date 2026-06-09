"""
Task 8 — PageIndex Vectorless RAG (Nâng cấp toàn diện lên Chat Completion API).

Đăng ký tài khoản tại: https://pageindex.ai/
Tài liệu hướng dẫn mới: https://docs.pageindex.ai/endpoints#-chat-api-beta
"""

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

# 📂 Cấu hình hệ thống thư mục dữ liệu của bạn
BASE_DIR = Path(__file__).parent.parent
DOC_DIR = BASE_DIR / "data" / "landing" / "legal"
PDF_DIR = BASE_DIR / "data" / "converted_pdfs"
META_FILE = BASE_DIR / "data" / "pageindex_meta.json"


def convert_docs_to_pdfs():
    """
    Tự động chuyển đổi .docx/.doc sang .pdf sử dụng thư viện aspose-words.
    """
    if not DOC_DIR.exists():
        return False
    doc_files = list(DOC_DIR.glob("*.docx")) + list(DOC_DIR.glob("*.doc"))
    if not doc_files:
        return False
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import aspose.words as aw
        for doc_file in doc_files:
            pdf_file_name = doc_file.stem + ".pdf"
            if (PDF_DIR / pdf_file_name).exists():
                continue
            doc = aw.Document(str(doc_file.resolve()))
            doc.save(str((PDF_DIR / pdf_file_name).resolve()))
        return True
    except Exception:
        return False


def get_client():
    """
    Khởi tạo PageIndexClient.
    """
    try:
        from pageindex import PageIndexClient
        if not PAGEINDEX_API_KEY:
            print("⚠ PAGEINDEX_API_KEY trống. Bỏ qua PageIndex fallback.")
            return None
        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except ImportError:
        print("❌ Lỗi: Chưa cài đặt thư viện pageindex. Hãy chạy: pip install pageindex")
        return None


def upload_documents():
    """
    Xử lý chuyển đổi Word -> PDF và đồng bộ lên PageIndex Cloud.
    """
    pi_client = get_client()
    if not pi_client:
        return

    convert_docs_to_pdfs()
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ Không có file PDF nào sẵn sàng.")
        return

    doc_mapping = {}
    if META_FILE.exists():
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                doc_mapping = json.load(f)
        except:
            doc_mapping = {}

    files_to_upload = [f for f in pdf_files if f.name not in doc_mapping]
    if not files_to_upload:
        print("✓ Tất cả tài liệu đã được lập chỉ mục (Index) từ trước.")
        return

    print(f"-> Phát hiện {len(files_to_upload)} file mới. Tiến hành upload...")
    for pdf_file in files_to_upload:
        try:
            print(f"⏳ Đang upload: {pdf_file.name} ...")
            submit_result = pi_client.submit_document(str(pdf_file))
            doc_id = None
            if isinstance(submit_result, dict):
                doc_id = submit_result.get("doc_id")
            elif isinstance(submit_result, str):
                doc_id = submit_result
                
            if doc_id:
                doc_mapping[pdf_file.name] = doc_id
                print(f"   ✓ Thành công. Mã doc_id: {doc_id}")
                while True:
                    if pi_client.is_retrieval_ready(doc_id):
                        break
                    time.sleep(3)
        except Exception as e:
            print(f"   ✕ Lỗi: {str(e)}")

    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_mapping, f, ensure_ascii=False, indent=4)


def pageindex_search(query: str, top_k: int = 3) -> list[dict]:
    """
    Sử dụng Chat/Chat-Completion API mới của PageIndex để trích xuất tri thức
    Không phụ thuộc vào các hàm Retrieval cũ đã bị chặn (Deprecated).
    """
    pi_client = get_client()
    if not pi_client or not META_FILE.exists():
        return []

    with open(META_FILE, "r", encoding="utf-8") as f:
        doc_mapping = json.load(f)

    all_results = []

    # Câu lệnh hướng dẫn hệ thống hoạt động như một bộ trích xuất dữ liệu gốc
    system_instruction = (
        "Bạn là trợ lý trích xuất văn bản pháp luật. Hãy tìm kiếm các phân đoạn hoặc điều khoản "
        "liên quan mật thiết nhất đến câu hỏi của người dùng trong tài liệu được cung cấp. "
        "Hãy trích dẫn nguyên văn hoặc tóm tắt cực kỳ sát nghĩa đoạn văn bản đó. Không tự bịa đặt thông tin."
    )

    for filename, doc_id in doc_mapping.items():
        try:
            print(f"🔍 Đang truy vấn qua Chat API tới tài liệu [{filename}]...")
            
            # Khởi tạo danh sách tin nhắn theo chuẩn OpenAI/PageIndex Chat mới
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": query}
            ]
            
            # Gọi hàm Chat API (Thử nghiệm linh hoạt giữa .chat hoặc .chat_completion tùy thuộc phiên bản SDK)
            response = None
            if hasattr(pi_client, "chat_completion"):
                response = pi_client.chat_completion(doc_id=doc_id, messages=messages)
            elif hasattr(pi_client, "chat"):
                response = pi_client.chat(doc_id=doc_id, messages=messages)
            else:
                # Fallback nếu gọi thẳng qua hàm submit_query nhưng không dùng polling
                response = pi_client.submit_query(doc_id=doc_id, query=query)

            # Phân tách câu trả lời từ cấu trúc phản hồi Chat
            answer_text = ""
            if isinstance(response, dict):
                # Theo chuẩn Chat Completion của PageIndex
                choices = response.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    message_obj = choices[0].get("message", {})
                    answer_text = message_obj.get("content", "")
                else:
                    # Đề phòng cấu trúc phản hồi phẳng
                    answer_text = response.get("answer") or response.get("content") or response.get("response") or ""
            elif isinstance(response, str):
                answer_text = response

            if answer_text and answer_text.strip():
                all_results.append({
                    "content": answer_text.strip(),
                    "score": 1.0,  # Chat API mặc định trả về câu trả lời tối ưu nhất từ cây tri thức
                    "metadata": {
                        "filename": filename,
                        "node_title": "Chat Context Extraction"
                    },
                    "source": "pageindex_chat"
                })

        except Exception as e:
            print(f"⚠ Lỗi khi tương tác Chat API trên file {filename}: {str(e)}")

    # Trả về danh sách kết quả phù hợp nhất
    return all_results[:top_k]


if __name__ == "__main__":
    print("=== MODULE VECTORLESS RAG (WORD DOC SUPPORT) ===")
    
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
    else:
        print("[1] Tiến trình xử lý dữ liệu và Upload...")
        upload_documents()

        print("\n[2] Test thử nghiệm câu truy vấn:")
        # Thay đổi từ khóa thành các câu hỏi thực tế về văn bản của bạn để đạt độ chính xác cao nhất
        query_test = "Nội dung chính của văn bản này quy định về vấn đề gì?"
        results = pageindex_search(query_test, top_k=3)
        
        if not results:
            print("Không tìm thấy dữ liệu phù hợp từ hệ thống Chat API PageIndex.")
        else:
            print(f"\n✨ Tìm thấy {len(results)} câu trả lời trích xuất từ tài liệu:")
            for idx, r in enumerate(results, 1):
                print(f"[{idx}] [File: {r['metadata']['filename']}]")
                print(f"    Nội dung trích xuất:\n    {r['content']}\n")
                print("-" * 60)
