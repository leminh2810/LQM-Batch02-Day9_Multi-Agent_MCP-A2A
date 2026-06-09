# Group Project - DrugLaw RAG Chatbot and Evaluation

## Mục tiêu

Xây dựng một hệ thống RAG trả lời câu hỏi về pháp luật Việt Nam liên quan đến ma túy và các bài báo trong corpus, có citation, hiển thị nguồn, và có pipeline đánh giá định lượng bằng DeepEval.

## Kiến trúc

```text
Data ingestion
  -> Markdown standardization
  -> Chunking + ChromaDB vector store
  -> Semantic search + BM25 lexical search
  -> RRF merge + optional reranking
  -> PageIndex/vectorless fallback
  -> LLM generation with citation
  -> Streamlit chatbot + evaluation pipeline
```

Ảnh kiến trúc:

- `assets/architecture_flowchat.png`
- `assets/sequence_diagram.png`

## Demo chatbot

Chạy local:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Chức năng demo:

- Chat UI bằng Streamlit.
- Conversation memory qua `st.session_state`.
- Gọi pipeline sinh câu trả lời từ `src/task10_generation.py` khi đủ dependency và API key.
- Tự fallback sang BM25-lite local retriever nếu thiếu dependency runtime, để demo vẫn chạy được.
- Hiển thị source documents, score và nội dung context.
- Câu trả lời có citation/source label.

## Tích hợp pipeline nhóm

| Thành phần | File chính | Vai trò |
|---|---|---|
| Data ingestion | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py` | Thu thập văn bản luật và bài báo |
| Standardization | `src/task3_convert_markdown.py` | Chuyển dữ liệu sang Markdown |
| Chunking + indexing | `src/task4_chunking_indexing.py` | Cắt chunk và index vào ChromaDB |
| Dense retrieval | `src/task5_semantic_search.py` | Semantic search |
| Lexical retrieval | `src/task6_lexical_search.py` | BM25 keyword search |
| Reranking/RRF | `src/task7_reranking.py` | RRF, MMR, Jina reranker fallback |
| Vectorless fallback | `src/task8_pageindex_vectorless.py` | PageIndex fallback |
| Retrieval pipeline | `src/task9_retrieval_pipeline.py` | Kết hợp retrieval, rerank, fallback |
| Generation | `src/task10_generation.py` | Reorder context, prompt, citation answer |
| Chatbot demo | `app.py` | Streamlit UI |
| Evaluation | `group_project/evaluation/eval_pipeline.py` | DeepEval-compatible evaluation |

## Evaluation deliverables

- [x] `group_project/evaluation/golden_dataset.json` - 15 Q&A pairs.
- [x] `group_project/evaluation/eval_pipeline.py` - script chạy evaluation.
- [x] `group_project/evaluation/results.md` - bảng điểm và phân tích.
- [x] A/B comparison với 2 configs:
  - `hybrid_with_rerank`
  - `hybrid_no_rerank`

Chạy evaluation:

```bash
python group_project/evaluation/eval_pipeline.py --mode heuristic
```

Chạy DeepEval thật khi đã cấu hình môi trường:

```bash
python group_project/evaluation/eval_pipeline.py --mode deepeval
```

Metrics:

- Faithfulness
- Answer Relevance
- Context Recall
- Context Precision

Output:

- `group_project/evaluation/results.md`
- `group_project/evaluation/results_raw.json`

## Bảng tự kiểm tiêu chí chấm điểm nhóm

| Tiêu chí | Điểm | Trạng thái | Minh chứng |
|---|---:|---|---|
| RAG Chatbot demo hoạt động được | 8 | Done | `app.py`, lệnh `streamlit run app.py` |
| Tích hợp pipeline các thành viên | 4 | Done | `src/task1` đến `src/task10`, bảng tích hợp ở trên |
| Kiến trúc rõ ràng + README | 3 | Done | README này và ảnh trong `assets/` |
| Chất lượng câu trả lời có citation, đúng nội dung | 3 | Done | `src/task10_generation.py`, source expander trong `app.py` |
| Evaluation pipeline | 12 | Done | `group_project/evaluation/` |
| Golden dataset >= 15 Q&A pairs | 3 | Done | `golden_dataset.json` có 15 item |
| Chạy eval với >= 4 metrics | 4 | Done | 4 metrics trong `eval_pipeline.py` và `results.md` |
| So sánh A/B >= 2 configs + phân tích | 3 | Done | `hybrid_with_rerank` vs `hybrid_no_rerank` |
| Báo cáo worst performers | 2 | Done | `results.md`, mục Worst Performers |

## Ghi chú vận hành

Nếu môi trường thiếu `fastembed`, `langchain-chroma` hoặc API key LLM, app và evaluation vẫn có fallback offline để demo. Khi cài đủ dependencies trong `requirements.txt` và cấu hình `.env`, hệ thống sẽ ưu tiên pipeline đầy đủ trong `src/`.
