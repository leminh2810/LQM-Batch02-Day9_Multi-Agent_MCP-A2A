import os
import json
from pathlib import Path
from markitdown import MarkItDown

# Định nghĩa các thư mục gốc
LANDING_DIR = Path(__file__).parent / "data" / "landing"
STANDARDIZED_DIR = Path(__file__).parent / "data" / "standardized"


def convert_all_to_markdown():
    # 1. Khởi tạo công cụ MarkItDown của Microsoft
    md = MarkItDown()

    if not LANDING_DIR.exists():
        print(f"⚠ Thư mục nguồn không tồn tại: {LANDING_DIR}")
        return

    print("=== BẮT ĐẦU CHUYỂN ĐỔI SANG MARKDOWN ===")

    # 2. Sử dụng os.walk để quét qua tất cả các file và thư mục con một cách tự động
    for root, dirs, files in os.walk(LANDING_DIR):
        for file in files:
            file_path = Path(root) / file
            
            # Bỏ qua các file ẩn hệ thống (như .DS_Store trên Mac)
            if file.startswith("."):
                continue

            # 3. Tính toán cấu trúc thư mục con tương ứng (ví dụ: legal/ hoặc news/)
            relative_path = file_path.relative_to(LANDING_DIR)
            sub_dir = relative_path.parent
            
            # Tạo thư mục đích tương ứng trong data/standardized/ nếu chưa tồn tại
            target_dir = STANDARDIZED_DIR / sub_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            # Định nghĩa tên file đầu ra (.md)
            output_filename = f"{file_path.stem}.md"
            output_path = target_dir / output_filename

            print(f"-> Đang xử lý: {relative_path}")

            try:
                # 4. Xử lý riêng biệt cho file dữ liệu JSON (từ Task 2) và tài liệu văn bản thông thường
                if file_path.suffix.lower() == ".json":
                    # Nếu là file bài báo JSON, ta bóc tách nội dung đã crawl để ghi đẹp hơn
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Tạo nội dung Markdown có kèm tiêu đề và meta
                    title = data.get("title", "Không rõ tiêu đề")
                    url = data.get("url", "#")
                    date = data.get("date_crawled", "")
                    content = data.get("content_markdown", "")
                    
                    markdown_content = f"# {title}\n\n"
                    markdown_content += f"- **Nguồn:** [{url}]({url})\n"
                    markdown_content += f"- **Ngày cào:** {date}\n\n"
                    markdown_content += f"---\n\n{content}"
                
                else:
                    # Nếu là PDF, DOCX, XLSX, PPTX, TXT... Sử dụng sức mạnh của MarkItDown
                    result = md.convert(str(file_path))
                    markdown_content = result.text_content

                # 5. Lưu nội dung Markdown xuống thư mục đích
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)

                print(f"   ✓ Thành công -> Saved: data/standardized/{sub_dir}/{output_filename}")

            except Exception as e:
                print(f"   ❌ Thất bại khi chuyển đổi file {file}: {str(e)}")

    print("=== HOÀN THÀNH QUÁ TRÌNH CHUYỂN ĐỔI ===")


if __name__ == "__main__":
    convert_all_to_markdown()