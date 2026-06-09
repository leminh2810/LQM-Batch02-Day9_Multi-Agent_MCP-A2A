"""
Multi-agent RAG orchestration for the DrugLaw chatbot.

The agents are intentionally lightweight Python components so the project can
run in the same classroom environment as the existing Task 9/10 pipeline.  The
orchestrator keeps a trace for observability and falls back to an extractive
answer when an OpenAI API key is not configured.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .task9_retrieval_pipeline import retrieve
    from .task10_generation import (
        SYSTEM_PROMPT,
        TEMPERATURE,
        TOP_K,
        TOP_P,
        format_context,
        reorder_for_llm,
    )
except ImportError as exc:
    if getattr(exc, "name", None):
        raise
    from task9_retrieval_pipeline import retrieve
    from task10_generation import (
        SYSTEM_PROMPT,
        TEMPERATURE,
        TOP_K,
        TOP_P,
        format_context,
        reorder_for_llm,
    )


load_dotenv()


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Tôi không thể xác minh thông tin này từ nguồn hiện có."
)


@dataclass(frozen=True)
class AgentTrace:
    agent: str
    action: str
    output: dict[str, Any]


@dataclass(frozen=True)
class MultiAgentConfig:
    top_k: int = TOP_K
    score_threshold: float = 0.3
    use_reranking: bool = True
    model: str = "gpt-4o-mini"


class QueryPlannerAgent:
    """Classify the query and choose retrieval hints for downstream agents."""

    LEGAL_TERMS = {
        "luật",
        "điều",
        "nghị định",
        "nghị quyết",
        "quốc hội",
        "hình phạt",
        "xử phạt",
        "cai nghiện",
        "ma túy",
    }
    NEWS_TERMS = {"nghệ sĩ", "bị bắt", "tin", "bài báo", "năm 2024", "sự kiện"}

    def run(self, query: str, config: MultiAgentConfig) -> tuple[dict[str, Any], AgentTrace]:
        normalized = query.lower()
        legal_hits = [term for term in self.LEGAL_TERMS if term in normalized]
        news_hits = [term for term in self.NEWS_TERMS if term in normalized]

        if legal_hits and not news_hits:
            intent = "legal"
        elif news_hits and not legal_hits:
            intent = "news"
        elif legal_hits and news_hits:
            intent = "mixed"
        else:
            intent = "general"

        plan = {
            "intent": intent,
            "query": query.strip(),
            "expanded_top_k": min(max(config.top_k + 2, config.top_k), 10),
            "score_threshold": config.score_threshold,
            "use_reranking": config.use_reranking,
            "keywords": sorted(set(legal_hits + news_hits)),
        }
        return plan, AgentTrace("query_planner", "analyze_query", plan)


class RetrievalAgent:
    """Call the existing hybrid retrieval pipeline."""

    def run(self, plan: dict[str, Any]) -> tuple[list[dict], AgentTrace]:
        chunks = retrieve(
            plan["query"],
            top_k=plan["expanded_top_k"],
            score_threshold=plan["score_threshold"],
            use_reranking=plan["use_reranking"],
        )
        output = {
            "retrieved": len(chunks),
            "source": chunks[0].get("source", "none") if chunks else "none",
            "top_score": chunks[0].get("score", 0.0) if chunks else 0.0,
        }
        return chunks, AgentTrace("retrieval_agent", "hybrid_retrieve", output)


class EvidenceCuratorAgent:
    """Deduplicate, cap, and reorder evidence before generation."""

    def run(
        self,
        chunks: list[dict],
        config: MultiAgentConfig,
    ) -> tuple[list[dict], AgentTrace]:
        unique_chunks: list[dict] = []
        seen = set()
        for chunk in chunks:
            content = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
            metadata = chunk.get("metadata", {})
            identity = (
                metadata.get("filename") or metadata.get("source") or "unknown",
                metadata.get("chunk_index"),
                content[:160],
            )
            if content and identity not in seen:
                seen.add(identity)
                unique_chunks.append(chunk)

        selected = reorder_for_llm(unique_chunks[: config.top_k])
        output = {
            "input_chunks": len(chunks),
            "selected_chunks": len(selected),
            "deduplicated": len(chunks) - len(unique_chunks),
        }
        return selected, AgentTrace("evidence_curator", "dedupe_and_reorder", output)


class ReasoningAgent:
    """Build a concise response strategy from the query and curated evidence."""

    def run(
        self,
        query: str,
        plan: dict[str, Any],
        chunks: list[dict],
    ) -> tuple[dict[str, Any], AgentTrace]:
        query_terms = _tokenize(query)
        evidence_notes = []
        total_overlap = 0

        for index, chunk in enumerate(chunks, 1):
            content = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
            metadata = chunk.get("metadata", {})
            source = metadata.get("filename") or metadata.get("source") or f"Document_{index}"
            overlap = len(query_terms & _tokenize(content[:1200]))
            total_overlap += overlap
            if overlap > 0:
                evidence_notes.append(
                    {
                        "source": source,
                        "overlap": overlap,
                        "score": chunk.get("score", 0.0),
                    }
                )

        intent = plan.get("intent", "general")
        if intent == "legal":
            response_style = "explain_rules_then_apply_to_question"
            focus = "Nêu quy định, điều kiện áp dụng, mức xử lý và giới hạn thông tin theo nguồn."
        elif intent == "news":
            response_style = "summarize_events_with_entities_and_dates"
            focus = "Tóm tắt sự kiện, nhân vật/tổ chức, thời điểm và nguồn bài viết."
        elif intent == "mixed":
            response_style = "separate_legal_rules_and_news_facts"
            focus = "Tách rõ phần quy định pháp luật và phần sự kiện báo chí."
        else:
            response_style = "direct_answer_with_context_limits"
            focus = "Trả lời trực tiếp, chỉ dùng nội dung được truy xuất."

        sufficient_evidence = bool(chunks) and (total_overlap > 0 or max(_scores(chunks), default=0.0) >= 0.35)
        reasoning = {
            "intent": intent,
            "response_style": response_style,
            "focus": focus,
            "sufficient_evidence": sufficient_evidence,
            "must_cite": True,
            "avoid": [
                "không dùng kiến thức ngoài context",
                "không suy đoán nếu thiếu chứng cứ",
                "không trộn lẫn nguồn pháp luật và nguồn báo chí",
            ],
            "evidence_notes": sorted(
                evidence_notes,
                key=lambda item: (item["overlap"], item["score"]),
                reverse=True,
            )[:3],
        }
        trace_output = {
            "intent": reasoning["intent"],
            "response_style": reasoning["response_style"],
            "sufficient_evidence": reasoning["sufficient_evidence"],
            "top_evidence": reasoning["evidence_notes"],
        }
        return reasoning, AgentTrace("reasoning_agent", "build_response_strategy", trace_output)


class AnswerWriterAgent:
    """Generate the final answer, using an extractive fallback when needed."""

    def run(
        self,
        query: str,
        chunks: list[dict],
        config: MultiAgentConfig,
        reasoning: dict[str, Any] | None = None,
    ) -> tuple[str, AgentTrace]:
        reasoning = reasoning or {}
        if not chunks:
            trace = AgentTrace(
                "answer_writer",
                "no_context",
                {"mode": "insufficient_evidence"},
            )
            return INSUFFICIENT_EVIDENCE_MESSAGE, trace

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            answer = self._extractive_answer(query, chunks, reasoning=reasoning)
            trace = AgentTrace(
                "answer_writer",
                "extractive_fallback",
                {
                    "mode": "offline",
                    "reason": "missing_OPENAI_API_KEY",
                    "used_reasoning": bool(reasoning),
                },
            )
            return answer, trace

        context = format_context(chunks)
        reasoning_block = self._format_reasoning_guidance(reasoning)
        user_message = (
            "Dưới đây là các tài liệu ngữ cảnh được trích xuất bởi hệ thống "
            "multi-agent:\n"
            "=================== NGỮ CẢNH TÀI LIỆU ===================\n"
            f"{context}\n"
            "=========================================================\n\n"
            "Reasoning agent guidance for the final answer:\n"
            f"{reasoning_block}\n\n"
            "Dựa hoàn toàn vào ngữ cảnh trên, hãy trả lời câu hỏi sau:\n"
            f"Question: {query}"
        )

        client = OpenAI(api_key=api_key)
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or INSUFFICIENT_EVIDENCE_MESSAGE
            trace = AgentTrace(
                "answer_writer",
                "llm_generate",
                {"mode": "openai", "model": config.model},
            )
            return answer, trace
        except Exception as exc:
            answer = self._extractive_answer(query, chunks, reasoning=reasoning)
            trace = AgentTrace(
                "answer_writer",
                "extractive_fallback",
                {"mode": "offline", "reason": str(exc), "used_reasoning": bool(reasoning)},
            )
            return answer, trace

    def _format_reasoning_guidance(self, reasoning: dict[str, Any]) -> str:
        if not reasoning:
            return "- Answer directly from context, with citations."
        avoid = "; ".join(reasoning.get("avoid", []))
        evidence = "; ".join(
            f"{item.get('source')} score={item.get('score')} overlap={item.get('overlap')}"
            for item in reasoning.get("evidence_notes", [])
        )
        return (
            f"- Intent: {reasoning.get('intent', 'general')}\n"
            f"- Response style: {reasoning.get('response_style', 'direct_answer_with_context_limits')}\n"
            f"- Focus: {reasoning.get('focus', 'Answer directly from retrieved context.')}\n"
            f"- Evidence sufficient: {reasoning.get('sufficient_evidence', False)}\n"
            f"- Avoid: {avoid or 'do not guess'}\n"
            f"- Strong evidence: {evidence or 'none'}\n"
            "- Do not reveal internal reasoning. Use it only to organize the final answer."
        )

    def _extractive_answer(
        self,
        query: str,
        chunks: list[dict],
        reasoning: dict[str, Any] | None = None,
        max_chars: int = 1200,
    ) -> str:
        reasoning = reasoning or {}
        query_terms = _tokenize(query)
        candidates: list[tuple[int, int, str, str]] = []

        for index, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("filename") or metadata.get("source") or f"Document_{index}"
            sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.get("content", ""))
            for position, sentence in enumerate(sentences):
                clean_sentence = re.sub(r"\s+", " ", sentence).strip()
                if not clean_sentence:
                    continue
                overlap = len(query_terms & _tokenize(clean_sentence))
                candidates.append((overlap, -position, clean_sentence, source))

        ranked = sorted(candidates, reverse=True)
        selected = [
            f"{sentence} [{source}]"
            for overlap, _, sentence, source in ranked[:3]
            if overlap > 0
        ]
        if not selected and chunks:
            metadata = chunks[0].get("metadata", {})
            source = metadata.get("filename") or metadata.get("source") or "Document_1"
            selected = [f"{chunks[0].get('content', '').strip()[:max_chars]} [{source}]"]

        answer = " ".join(selected).strip()
        if answer and reasoning:
            prefix = _answer_prefix(reasoning)
            if prefix:
                answer = f"{prefix} {answer}"
        return answer[:max_chars] if answer else INSUFFICIENT_EVIDENCE_MESSAGE


class VerificationAgent:
    """Perform a deterministic citation and evidence sanity check."""

    CITATION_PATTERN = re.compile(r"\[[^\]]+\]")

    def run(self, answer: str, chunks: list[dict]) -> tuple[dict[str, Any], AgentTrace]:
        citations = self.CITATION_PATTERN.findall(answer)
        has_context = bool(chunks)
        says_insufficient = INSUFFICIENT_EVIDENCE_MESSAGE in answer
        passed = bool(citations and has_context) or says_insufficient

        verification = {
            "passed": passed,
            "citation_count": len(citations),
            "has_context": has_context,
            "says_insufficient": says_insufficient,
        }
        return verification, AgentTrace("verification_agent", "check_citations", verification)


class MultiAgentRAGSystem:
    """Coordinate planner, retriever, curator, writer, and verifier agents."""

    def __init__(self, config: MultiAgentConfig | None = None):
        self.config = config or MultiAgentConfig()
        self.planner = QueryPlannerAgent()
        self.retriever = RetrievalAgent()
        self.curator = EvidenceCuratorAgent()
        self.reasoner = ReasoningAgent()
        self.writer = AnswerWriterAgent()
        self.verifier = VerificationAgent()

    def run(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        config = self.config
        if top_k is not None:
            config = MultiAgentConfig(
                top_k=top_k,
                score_threshold=config.score_threshold,
                use_reranking=config.use_reranking,
                model=config.model,
            )

        trace: list[AgentTrace] = []

        plan, step = self.planner.run(query, config)
        trace.append(step)

        retrieved_chunks, step = self.retriever.run(plan)
        trace.append(step)

        curated_chunks, step = self.curator.run(retrieved_chunks, config)
        trace.append(step)

        reasoning, step = self.reasoner.run(query, plan, curated_chunks)
        trace.append(step)

        answer, step = self.writer.run(query, curated_chunks, config, reasoning=reasoning)
        trace.append(step)

        verification, step = self.verifier.run(answer, curated_chunks)
        trace.append(step)

        if not verification["passed"]:
            answer = (
                f"{answer}\n\n"
                "Lưu ý: hệ thống chưa xác minh đủ citation trong câu trả lời."
            )

        return {
            "answer": answer,
            "sources": curated_chunks,
            "retrieval_source": (
                retrieved_chunks[0].get("source", "none") if retrieved_chunks else "none"
            ),
            "plan": plan,
            "reasoning": reasoning,
            "verification": verification,
            "agent_trace": [asdict(item) for item in trace],
        }


def run_multiagent_rag(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Convenience function compatible with the Streamlit backend contract."""

    return MultiAgentRAGSystem().run(query, top_k=top_k)


def _scores(chunks: list[dict]) -> list[float]:
    scores: list[float] = []
    for chunk in chunks:
        try:
            scores.append(float(chunk.get("score", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
    return scores


def _answer_prefix(reasoning: dict[str, Any]) -> str:
    if not reasoning.get("sufficient_evidence", False):
        return "Dựa trên nguồn hiện có, thông tin còn hạn chế."

    intent = reasoning.get("intent")
    if intent == "legal":
        return "Theo các nguồn pháp lý được truy xuất,"
    if intent == "news":
        return "Theo các bài viết trong corpus,"
    if intent == "mixed":
        return "Tách theo phần pháp lý và phần sự kiện,"
    return "Dựa trên các nguồn được truy xuất,"


def _tokenize(text: str) -> set[str]:
    normalized = text.lower()
    normalized = re.sub(r"[^\w\sÀ-ỹ]", " ", normalized, flags=re.UNICODE)
    return {token for token in normalized.split() if len(token) > 1}
