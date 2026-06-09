import asyncio
import json
from datetime import datetime
from pathlib import Path

# Import các thành phần cấu hình từ crawl4ai
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Định nghĩa đường dẫn lưu file: data/landing/news/
DATA_DIR = Path(__file__).parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách tối thiểu 5 bài báo thật về nghệ sĩ, người nổi tiếng Việt Nam liên quan tới ma tuý
ARTICLE_URLS = [
    "https://tuoitre.vn/nguoi-mau-an-tay-ca-si-chi-dan-bi-tam-giu-vi-lien-quan-den-ma-tuy-20241114101905096.htm",
    "https://thanhnien.vn/tu-vu-an-tay-chi-dan-nguoi-noi-tieng-choi-ma-tuy-se-tra-gia-rat-dat-185241117195843405.htm",
    "https://vietnamnet.vn/dien-vien-huu-tin-bi-tuyen-phat-7-nam-6-thang-tu-vi-to-chuc-su-dung-ma-tuy-2148782.html",
    "https://vnexpress.net/ca-si-chu-bin-bi-tam-giu-vi-to-chuc-su-dung-ma-tuy-4754593.html",
    "https://plo.vn/loat-be-boi-cua-nghe-si-viet-lien-quan-den-ma-tuy-post820251.html"
]


async def crawl_article(crawler: AsyncWebCrawler, url: str, config: CrawlerRunConfig) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.
    Truyền crawler instance vào để tối ưu kết nối, tránh khởi tạo lại nhiều lần.
    """
    print(f"-> Đang cào dữ liệu: {url}")
    result = await crawler.arun(url=url, config=config)
    
    title = "Unknown"
    content = ""
    
    if result.success:
        content = result.markdown
        if result.metadata and isinstance(result.metadata, dict):
            # Lấy tiêu đề từ metadata meta tags hoặc fallback về tag title
            title = result.metadata.get("title") or result.metadata.get("og:title") or "Unknown"
        
        # Phòng hờ trường hợp trang web chặn trả về trang 403 mặc dù success=True
        if "403" in title or "Forbidden" in title:
            title = "Lỗi kết nối (Bị chặn 403)"
    else:
        title = f"Lỗi cào dữ liệu: {result.error_message}"
        content = f"Không thể lấy nội dung từ link này. Lỗi: {result.error_message}"

    return {
        "url": url,
        "title": title.strip(),
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


async def crawl_all():
    """Crawl song song toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    # 1. Cấu hình Trình duyệt giả lập người dùng thật để tránh 403
    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-blink-features=AutomationControlled"],
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/"
        }
    )

    # 2. Bỏ qua cache để luôn lấy dữ liệu mới nhất
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
    )

    # 3. Sử dụng duy nhất một Session để cào đồng thời tất cả các link (Tăng tốc độ)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Tạo danh sách các task cần chạy song song
        tasks = [crawl_article(crawler, url, run_config) for url in ARTICLE_URLS]
        
        # Đợi tất cả các task hoàn thành
        results = await asyncio.gather(*tasks)

        # 4. Ghi dữ liệu ra file JSON
        for i, article in enumerate(results, 1):
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(article, f, ensure_ascii=False, indent=2)
                
            print(f" ✓ Đã lưu thành công: {filepath}")


if __name__ == "__main__":
    # Khởi chạy hàm async chính
    asyncio.run(crawl_all())