# Lab Solution - Legal Multi-Agent A2A Codelab

File này ghi lại các phần đã thực hiện theo `CODELAB.md`, gồm thay đổi code, cách test, lỗi đã gặp và hướng xử lý.

## Tổng quan

Dự án đã được hoàn thiện qua 5 stage chính:

1. Stage 1 - Direct LLM Calling
2. Stage 2 - LLM + RAG & Tools
3. Stage 3 - Single Agent với ReAct
4. Stage 4 - Multi-Agent In-Process
5. Stage 5 - Distributed A2A System

Ngoài nội dung codelab, đã bổ sung dashboard tương tác để người dùng trực quan hóa flow agent, start/stop service, gửi câu hỏi, xem log và đo latency.

## Stage 1 - Direct LLM Calling

### Đã phân tích

File chính:

- `stages/stage_1_direct_llm/main.py`
- `common/llm.py`

LLM được khởi tạo thông qua hàm `get_llm()` trong `common/llm.py`. Hàm này đọc cấu hình từ `.env`, gồm:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`

Sau đó tạo `ChatOpenAI` dùng OpenRouter API.

Message gửi đến LLM gồm:

- `SystemMessage`: định nghĩa vai trò/hành vi của model, ví dụ chuyên gia pháp lý.
- `HumanMessage`: chứa câu hỏi thật của người dùng.

Lý do cần cả hai:

- `SystemMessage` giúp định hướng cách trả lời, domain, tone và giới hạn hành vi.
- `HumanMessage` là input cụ thể cần xử lý.

### Bài tập 1.1 - Thay đổi câu hỏi

Đã sửa biến `QUESTION` trong:

- `stages/stage_1_direct_llm/main.py`

Câu hỏi mới dùng chủ đề pháp lý về lao động:

```python
QUESTION = (
    "Nguoi lao dong bi don phuong cham dut hop dong trai phap luat "
    "co the yeu cau boi thuong gi?"
)
```

### Bài tập 1.2 - Thêm temperature control

Đã sửa `common/llm.py`, thêm:

```python
temperature=0.3
```

Mục đích: làm output ổn định hơn, ít ngẫu nhiên hơn.

Ngoài ra đã thêm `.strip()` khi đọc env để tránh lỗi API key/model có khoảng trắng hoặc ký tự xuống dòng.

## Stage 2 - LLM + RAG & Tools

### File chính

- `stages/stage_2_rag_tools/main.py`

### Bài tập 2.1 - Thêm knowledge base entry

Đã thêm entry mới vào `LEGAL_KNOWLEDGE` với id:

```python
"labor_law"
```

Nội dung entry nói về Bộ luật Lao động Việt Nam 2019 và các trường hợp người sử dụng lao động có thể đơn phương chấm dứt hợp đồng.

Keywords đã thêm:

```python
["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"]
```

### Bài tập 2.2 - Tạo tool mới

Đã thêm tool:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    ...
```

Tool này trả về thời hiệu khởi kiện theo loại vụ án:

- `contract`: 4 năm
- `tort`: 2-3 năm tùy bang
- `property`: 5 năm

Đã thêm tool này vào danh sách `TOOLS`.

### Fix thêm để chạy ổn định

Trong Stage 2 đã gặp lỗi token/quota kiểu:

```text
You requested up to 700 tokens, but can only afford ...
```

Đã xử lý bằng cách giảm `max_tokens` trong `common/llm.py`.

Ngoài ra đã cải thiện matching trong `search_legal_database` để hỗ trợ keyword tiếng Việt tốt hơn.

## Stage 3 - Single Agent với ReAct

### File chính

- `stages/stage_3_single_agent/main.py`

### Đã phân tích

Stage 3 dùng:

```python
create_react_agent()
```

Khác với Stage 2:

- Stage 2 có manual tool loop.
- Stage 3 để ReAct agent tự quyết định gọi tool nào, gọi khi nào và tổng hợp kết quả.

### Bài tập 3.1 - Thêm tool tra cứu án lệ

Đã thêm tool:

```python
@tool
def search_case_law(keywords: str) -> str:
    ...
```

Tool hỗ trợ các keyword:

- `breach`
- `negligence`
- `contract`

Ví dụ kết quả:

```text
Hadley v. Baxendale (1854) - Consequential damages
```

Đã thêm tool vào danh sách `TOOLS`.

### Bài tập 3.2 - Debug agent reasoning

Code hiện tại dùng stream updates để quan sát reasoning/tool calls của agent thay vì chỉ gọi một lần rồi đợi kết quả.

## Stage 4 - Multi-Agent In-Process

### File chính

- `stages/stage_4_milti_agent/main.py`

### Đã phân tích

Các thành phần chính:

- `State` / `LegalState`: shared state giữa các node.
- `law_agent`: phân tích pháp lý chính.
- `tax_agent`: phân tích thuế.
- `compliance_agent`: phân tích compliance.
- `Send()`: dispatch nhiều task song song.
- `graph.add_node()` và `graph.add_edge()`: định nghĩa graph.

### Bước vẽ graph

Đã thêm chức năng vẽ graph bằng Mermaid/PNG thông qua helper trong Stage 4.

Có thể chạy mode vẽ graph bằng tham số CLI đã bổ sung.

### Bài tập 4.1 - Thêm privacy_agent

Đã thêm `privacy_agent` chuyên về GDPR và privacy law trong Stage 4.

Agent này nhận:

- câu hỏi gốc
- phân tích pháp lý từ `law_agent`

Sau đó trả về:

```python
{"privacy_analysis": response.content}
```

Đã thêm node `privacy_agent` vào graph và nối về bước aggregate.

### Bài tập 4.2 - Conditional routing

Đã sửa routing để chỉ gọi các agent chuyên biệt khi câu hỏi có keyword phù hợp:

- Tax: `tax`, `irs`, `thuế`
- Compliance: `compliance`, `sec`, `regulation`, `sox`, `aml`, `fcpa`
- Privacy: `data`, `privacy`, `gdpr`, `dữ liệu`

Nếu không có specialist nào phù hợp, graph đi thẳng tới aggregate.

## Stage 5 - Distributed A2A System

### Các service chính

| Service | Port | Vai trò |
|---|---:|---|
| Registry | 10000 | Service discovery |
| Customer Agent | 10100 | Entry point |
| Law Agent | 10101 | Orchestrator |
| Tax Agent | 10102 | Tax specialist |
| Compliance Agent | 10103 | Compliance specialist |

### Script chạy trên Windows

Đã tạo:

- `start_all.ps1`

Dùng để start toàn bộ service trên Windows PowerShell.

Lệnh chạy:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_all.ps1
```

### Test hệ thống

Test bằng:

```powershell
.\.venv\Scripts\python.exe test_client.py
```

Hệ thống đã chạy thành công:

- Client kết nối Customer Agent.
- Customer Agent delegate sang Law Agent.
- Law Agent gọi Tax Agent và Compliance Agent.
- Kết quả được aggregate và trả về client.

### Bài tập 5.1 - Trace request flow

Đã theo dõi `trace_id` trong logs để xem request đi qua các agent.

Đã tạo sequence diagram Mermaid:

- `sequence_diagram.mmd`

Lưu ý: Mermaid file phải bắt đầu trực tiếp bằng:

```text
sequenceDiagram
```

Không bọc bằng markdown fence ```mermaid, nếu không renderer có thể báo `UnknownDiagramError`.

### Bài tập 5.2 - Test dynamic discovery

Đã test bằng cách dừng Tax Agent rồi chạy lại `test_client.py`.

Kết quả quan sát:

- Registry vẫn trả endpoint Tax Agent vì registry lưu registration in-memory và chưa có health check.
- Law Agent gọi Tax Agent thì lỗi connection.
- Hệ thống bắt lỗi và tiếp tục xử lý các nhánh còn lại.
- Final response vẫn được aggregate với phần Tax bị thiếu/lỗi.

Điểm rút ra:

- Dynamic discovery hiện mới là discovery theo registration.
- Nếu muốn production-ready cần thêm health check hoặc TTL cho registration.

### Bài tập 5.3 - Modify Tax Agent behavior

Đã sửa prompt trong:

- `tax_agent/graph.py`

Mục tiêu:

- Tax Agent trả lời ngắn hơn.
- Giảm số token sinh ra.
- Giảm latency và chi phí.

Prompt đã yêu cầu Tax Agent trả về đúng 3 bullet ngắn và một disclaimer ngắn.

## Xử lý lỗi API/quota/model

Đã gặp các lỗi:

```text
402 Insufficient credits
401 User not found
```

Hướng xử lý:

- Kiểm tra API key trong `.env`.
- Đảm bảo key thuộc đúng account/org.
- Đổi sang model rẻ hơn:

```env
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Ngoài ra không nên paste API key vào chat/log public. Nếu key đã bị lộ, nên revoke và tạo key mới.

## Tối ưu latency

### Baseline

Trước tối ưu, latency end-to-end khoảng:

```text
45.915 giây
```

Nguyên nhân chính:

- Customer Agent gọi LLM để quyết định delegate.
- Law Agent tiếp tục gọi LLM.
- Tax/Compliance gọi LLM.
- Aggregate gọi LLM.
- Một số bước diễn ra tuần tự hoặc sinh output dài.

### Phương án tối ưu đã áp dụng

Đã sửa:

- `customer_agent/agent_executor.py`

Thêm direct delegation:

- Customer Agent bỏ vòng ReAct LLM trung gian.
- Customer Agent trực tiếp discover `legal_question`.
- Sau đó delegate thẳng sang Law Agent.

Biến điều khiển:

```env
CUSTOMER_DIRECT_DELEGATION=true
```

### Kết quả sau tối ưu

Sau tối ưu, latency đã đo được khoảng:

```text
25.339 giây
```

Một lượt test sau khi làm dashboard đo được khoảng:

```text
27.5 - 31.6 giây
```

Tùy thời điểm API và độ dài output, latency có dao động.

Mức giảm tốt nhất đã ghi nhận:

```text
45.915s -> 25.339s
giảm 20.576s, khoảng 44.8%
```

## Dashboard tương tác

Đã tạo dashboard để demo trực quan:

- `agent_dashboard.py`
- `agent_dashboard.html`
- `run_dashboard.ps1`
- `run_dashboard.bat`

### Chức năng thật trên dashboard

Dashboard hiện không chỉ là giao diện mô phỏng. Các control đã nối với backend thật:

- Hiển thị trạng thái port thật của 5 service.
- `Start services`: start các service thật.
- `Stop services`: dừng các service thật.
- Click từng dòng agent: start/stop riêng agent đó.
- Switch tối ưu latency: restart Customer Agent với `CUSTOMER_DIRECT_DELEGATION=true/false`.
- `Gửi Câu Hỏi`: chạy thật `test_client.py`.
- Live Trace Logs: đọc log thật từ thư mục `logs/`.
- Báo cáo cuối: lấy response thật từ lượt chạy.
- Hiển thị latency thực tế của lượt chạy.

### Cách chạy dashboard

Chạy:

```powershell
.\.venv\Scripts\python.exe agent_dashboard.py
```

Hoặc:

```powershell
.\run_dashboard.bat
```

Sau đó mở:

```text
http://127.0.0.1:8088
```

Không nên mở trực tiếp `agent_dashboard.html` bằng file hoặc Live Server nếu dashboard Python chưa chạy.

## Lệnh test nhanh

### Kiểm tra service ports

```powershell
netstat -ano | Select-String ':10000|:10100|:10101|:10102|:10103'
```

### Chạy full Stage 5

```powershell
.\.venv\Scripts\python.exe test_client.py
```

### Chạy dashboard

```powershell
.\.venv\Scripts\python.exe agent_dashboard.py
```

### Dừng dashboard nếu port 8088 bị chiếm

```powershell
netstat -ano | Select-String ':8088'
taskkill /PID <PID> /F
```

## Các file đã chỉnh/sinh thêm

### File code chính đã chỉnh

- `common/llm.py`
- `stages/stage_1_direct_llm/main.py`
- `stages/stage_2_rag_tools/main.py`
- `stages/stage_3_single_agent/main.py`
- `stages/stage_4_milti_agent/main.py`
- `customer_agent/agent_executor.py`
- `tax_agent/graph.py`
- `test_client.py`

### File mới

- `start_all.ps1`
- `run_dashboard.ps1`
- `run_dashboard.bat`
- `agent_dashboard.py`
- `agent_dashboard.html`
- `stage5_latency_demo.html`
- `sequence_diagram.mmd`
- `exercises/SOLUTIONS.md`
- `lap.solutuin.md`

## Kết luận

Các yêu cầu chính trong `CODELAB.md` đã được thực hiện:

- Stage 1: hiểu direct LLM, đổi câu hỏi, thêm temperature.
- Stage 2: thêm knowledge base và tool mới.
- Stage 3: thêm tool án lệ cho ReAct agent.
- Stage 4: thêm privacy agent và conditional routing.
- Stage 5: chạy distributed A2A, trace request, test fault tolerance, sửa Tax Agent prompt.
- Bổ sung dashboard để người dùng tương tác trực tiếp và quan sát hệ thống chạy thật.

