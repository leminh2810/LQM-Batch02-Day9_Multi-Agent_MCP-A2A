# RAG Evaluation Results

## Framework sử dụng

- Framework chọn: DeepEval
- Chế độ chạy: `heuristic`
- Golden dataset: 15 Q&A pairs
- Raw output: `results_raw.json`

## Overall Scores

| Metric | hybrid_with_rerank | hybrid_no_rerank | Delta |
|---|---:|---:|---:|
| faithfulness | 1.000 | 1.000 | +0.000 |
| answer_relevance | 0.641 | 0.623 | +0.018 |
| context_recall | 0.925 | 0.925 | +0.000 |
| context_precision | 0.545 | 0.539 | +0.005 |
| average | 0.778 | 0.772 | +0.006 |

## A/B Comparison Analysis

**Config A - hybrid_with_rerank:** Semantic search + BM25, merge bằng RRF, bật reranking.

**Config B - hybrid_no_rerank:** Semantic search + BM25, merge bằng RRF, tắt reranking.

**Kết luận:** `hybrid_with_rerank` đang có điểm trung bình tốt hơn. Nếu config rerank thắng, hệ thống đang hưởng lợi từ bước lọc lại top-k; nếu config không rerank thắng, cần kiểm tra chất lượng API reranker hoặc ngưỡng score.

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|---|---:|---:|---:|---:|---|---|
| 1 | Theo bài PLO, vụ Miu Lê được phát hiện ở đâu và khi nào? | 1.000 | 0.270 | 0.852 | 0.444 | Generation | Cần bổ sung expected_context chi tiết hơn hoặc cải thiện prompt. |
| 2 | Tội tổ chức sử dụng trái phép chất ma túy khác gì với hành vi chỉ sử dụng trái phép chất ... | 1.000 | 0.509 | 0.766 | 0.417 | Retrieval | Cần bổ sung expected_context chi tiết hơn hoặc cải thiện prompt. |
| 3 | Ai có trách nhiệm phòng, chống ma túy theo Luật Phòng, chống ma túy 2021? | 1.000 | 0.512 | 0.800 | 0.408 | Retrieval | Cần bổ sung expected_context chi tiết hơn hoặc cải thiện prompt. |

## Recommendations

### Cải tiến 1
**Action:** Chuẩn hóa lại dữ liệu crawl, loại bỏ menu/header/footer và sửa lỗi encoding trước khi chunking.
**Expected impact:** Tăng context precision vì top-k bớt nhiễu và chunk chứa nội dung chính rõ hơn.

### Cải tiến 2
**Action:** Thêm metadata filter theo loại tài liệu (`legal`, `news`) và nguồn (`PLO`, `Kenh14`, `aFamily`) dựa trên intent câu hỏi.
**Expected impact:** Tăng context recall cho câu hỏi pháp luật và câu hỏi tin tức có nguồn cụ thể.

### Cải tiến 3
**Action:** Tune `top_k`, RRF weight và bật reranker ổn định; log lại source score để debug từng query.
**Expected impact:** Cải thiện cả context recall lẫn faithfulness vì LLM nhận evidence đúng và ít mâu thuẫn hơn.
