"""
Streamlit demo for the group RAG chatbot.

The app uses the real Task 10 generation pipeline when all runtime dependencies
and API keys are available. If the local environment is missing optional RAG
dependencies, it falls back to the same BM25-lite retriever used by the
evaluation script so the demo remains runnable for grading.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_rag_backend():
    try:
        from src.multiagent_rag import run_multiagent_rag

        return "multiagent_rag_pipeline", run_multiagent_rag
    except ModuleNotFoundError:
        from group_project.evaluation.eval_pipeline import extractive_answer, fallback_retrieve

        def fallback_generate(query: str, top_k: int = 5) -> dict:
            chunks = fallback_retrieve(query, top_k=top_k, score_threshold=0.2, use_reranking=True)
            answer = extractive_answer(query, [chunk["content"] for chunk in chunks])
            if chunks:
                citations = []
                for chunk in chunks[:3]:
                    meta = chunk.get("metadata", {})
                    filename = meta.get("filename", "local_corpus")
                    chunk_index = meta.get("chunk_index", 0)
                    citations.append(f"[{filename}, chunk {chunk_index}]")
                answer = f"{answer}\n\nNguồn: {' '.join(citations)}"
            else:
                answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
            return {
                "answer": answer,
                "sources": chunks,
                "retrieval_source": "fallback_bm25_lite",
                "agent_trace": [],
            }

        return "fallback_bm25_lite", fallback_generate


def render_source(index: int, source: dict) -> None:
    metadata = source.get("metadata", {})
    filename = metadata.get("filename") or metadata.get("source") or f"source_{index}"
    score = source.get("score", 0.0)
    content = source.get("content", "").strip()
    with st.expander(f"{index}. {filename} · score {score:.3f}"):
        st.caption(metadata.get("source", "local"))
        st.write(content[:1400] if content else "Không có nội dung context.")

def clean_answer_text(answer: str) -> str:
    return re.sub(r"\n{2,}Ngu.+$", "", answer.strip(), flags=re.IGNORECASE | re.DOTALL)


def source_name(source: dict) -> str:
    metadata = source.get("metadata", {})
    filename = metadata.get("filename") or metadata.get("source") or "source"
    return Path(str(filename).replace("\\", "/")).name


def source_label(index: int, source: dict) -> str:
    filename = source_name(source)
    score = source.get("score", 0.0)
    return f"{index}. {filename} ({score:.2f})"


def render_source_line(sources: list[dict]) -> None:
    if not sources:
        return
    labels = []
    seen = set()
    for source in sources:
        label_key = source_name(source)
        if label_key in seen:
            continue
        seen.add(label_key)
        label = source_label(len(labels) + 1, source)
        labels.append(label)
        if len(labels) == 4:
            break
    extra = max(0, len({source_name(source) for source in sources}) - len(labels))
    suffix = f" và {extra} nguồn khác" if extra > 0 else ""
    st.caption(f"Nguồn: {' · '.join(labels)}{suffix}")


def render_answer(content: str, sources: list[dict]) -> None:
    st.markdown(clean_answer_text(content))
    render_source_line(sources)


def render_agent_trace(trace: list[dict]) -> None:
    if not trace:
        return
    with st.expander("Multi-agent trace"):
        for step in trace:
            agent = step.get("agent", "agent")
            action = step.get("action", "run")
            output = step.get("output", {})
            st.markdown(f"**{agent}** · `{action}`")
            st.json(output, expanded=False)


st.set_page_config(page_title="DrugLaw RAG Chatbot", layout="wide")
st.markdown(
    """
    <style>
    .stChatMessage p {
        line-height: 1.65;
    }
    .stChatMessage [data-testid="stCaptionContainer"] {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }
    .stChatMessage [data-testid="stExpander"] {
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

backend_name, generate_answer = load_rag_backend()

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Pháp luật ma túy và tin tức liên quan")
    top_k = st.slider("Top-k context", min_value=3, max_value=8, value=5, step=1)
    show_debug_trace = st.toggle("Hiện trace agent", value=False)
    st.info(f"Backend: `{backend_name}`")
    if backend_name in {"full_rag_pipeline", "multiagent_rag_pipeline"} and not os.getenv("OPENAI_API_KEY"):
        st.warning("Thiếu OPENAI_API_KEY, pipeline generation thật có thể trả về thông báo lỗi cấu hình.")
    if st.button("Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("DrugLaw RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bạn có thể hỏi về Luật Phòng, chống ma túy, xử lý hành vi sử dụng ma túy, hoặc các bài báo trong corpus.",
            "sources": [],
            "agent_trace": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(message["content"], message.get("sources", []))
            if show_debug_trace:
                render_agent_trace(message.get("agent_trace", []))
        else:
            st.write(message["content"])

prompt = st.chat_input("Nhập câu hỏi...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": [], "agent_trace": []})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất context và sinh câu trả lời..."):
            result = generate_answer(prompt, top_k=top_k)
        answer = result.get("answer", "Tôi không thể xác minh thông tin này từ nguồn hiện có.")
        sources = result.get("sources", [])
        agent_trace = result.get("agent_trace", [])
        render_answer(answer, sources)
        if show_debug_trace:
            render_agent_trace(agent_trace)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "agent_trace": agent_trace,
        }
    )
