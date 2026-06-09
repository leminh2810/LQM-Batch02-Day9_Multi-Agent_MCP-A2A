# Solutions

This file contains reference solutions for the exercises in this folder.

Use these after trying the exercises yourself.

---

## Exercise 2: Tools and Knowledge Base

Add the labor law entry to `LEGAL_KNOWLEDGE`:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

Create the new statute-of-limitations tool:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Add it to the tools list:

```python
tools = [search_legal_knowledge, check_statute_of_limitations]
```

Handle the tool call:

```python
if tool_call["name"] == "check_statute_of_limitations":
    tool_result = check_statute_of_limitations.invoke(tool_call["args"])
```

Run:

```bash
uv run python exercises/exercise_2_tools.py
```

---

## Exercise 4: Multi-Agent with Privacy Agent

Add routing for privacy questions:

```python
if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
    tasks.append(Send("privacy_agent", state))
```

Implement `privacy_agent`:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy, GDPR, data breach, quyền riêng tư,
nghĩa vụ thông báo vi phạm dữ liệu, và mức phạt tiềm năng.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Add privacy analysis to aggregation:

```python
if state.get("privacy_analysis"):
    sections.append(f"🔒 PHÂN TÍCH PRIVACY/GDPR:\n{state['privacy_analysis']}")
```

Add the node:

```python
graph.add_node("privacy_agent", privacy_agent)
```

Add the edge:

```python
graph.add_edge("privacy_agent", "aggregate_results")
```

Run:

```bash
uv run python exercises/exercise_4_multiagent.py
```
