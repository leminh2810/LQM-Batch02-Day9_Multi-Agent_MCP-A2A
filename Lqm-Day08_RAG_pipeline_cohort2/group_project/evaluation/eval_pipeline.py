"""
RAG Evaluation Pipeline for the group project.

Framework selected: DeepEval.

The script supports two execution modes:
    1. heuristic (default): deterministic offline scorer for classroom demos.
    2. deepeval: uses DeepEval metrics when the package and LLM credentials are ready.

Example:
    python group_project/evaluation/eval_pipeline.py --mode heuristic
    python group_project/evaluation/eval_pipeline.py --mode deepeval
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
RAW_RESULTS_PATH = Path(__file__).parent / "results_raw.json"


@dataclass(frozen=True)
class EvalConfig:
    name: str
    description: str
    use_reranking: bool
    top_k: int = 5
    score_threshold: float = 0.3


CONFIGS = [
    EvalConfig(
        name="hybrid_with_rerank",
        description="Semantic search + BM25, merge bằng RRF, bật reranking.",
        use_reranking=True,
    ),
    EvalConfig(
        name="hybrid_no_rerank",
        description="Semantic search + BM25, merge bằng RRF, tắt reranking.",
        use_reranking=False,
    ),
]


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) < 15:
        raise ValueError("Golden dataset must contain at least 15 Q&A pairs.")

    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(data, 1):
        missing = required - set(item)
        if missing:
            raise ValueError(f"Dataset item #{index} is missing fields: {sorted(missing)}")

    return data


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "the", "is", "are", "of", "to", "in",
        "là", "của", "và", "có", "theo", "những", "các", "một", "về",
        "cho", "trong", "được", "bị", "với", "nào", "gì", "khi", "đâu",
    }
    return {token for token in normalize_text(text).split() if token not in stopwords}


def overlap_score(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def harmonic_mean(values: list[float]) -> float:
    values = [max(v, 0.0001) for v in values]
    return len(values) / sum(1 / v for v in values)


def context_text(contexts: list[str]) -> str:
    return "\n".join(contexts)


def heuristic_metrics(
    question: str,
    answer: str,
    expected_answer: str,
    expected_context: str,
    retrieved_contexts: list[str],
) -> dict[str, float]:
    retrieved = context_text(retrieved_contexts)
    answer_expected = overlap_score(expected_answer, answer)
    answer_context = overlap_score(answer, retrieved)
    question_answer = overlap_score(question, answer)
    expected_context_hit = max(
        [overlap_score(expected_context, ctx) for ctx in retrieved_contexts] or [0.0]
    )
    useful_context_ratio = mean(
        [overlap_score(expected_answer, ctx) for ctx in retrieved_contexts] or [0.0]
    )

    return {
        "faithfulness": round(min(1.0, 0.35 + 0.65 * answer_context), 4),
        "answer_relevance": round(min(1.0, harmonic_mean([question_answer, answer_expected])), 4),
        "context_recall": round(min(1.0, max(expected_context_hit, overlap_score(expected_answer, retrieved))), 4),
        "context_precision": round(min(1.0, useful_context_ratio), 4),
    }


def import_rag_functions() -> tuple[Callable, Callable]:
    try:
        from src.task9_retrieval_pipeline import retrieve
        from src.task10_generation import generate_with_citation

        return generate_with_citation, retrieve
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        print(
            f"Warning: cannot import full RAG pipeline because `{missing}` is missing. "
            "Using local BM25-lite fallback retriever for evaluation."
        )
        return fallback_generate_with_citation, fallback_retrieve


def fallback_generate_with_citation(query: str, top_k: int = 5) -> dict:
    chunks = fallback_retrieve(query, top_k=top_k)
    return {
        "answer": extractive_answer(query, [chunk["content"] for chunk in chunks]),
        "sources": chunks,
        "retrieval_source": "fallback_bm25_lite",
    }


def load_local_corpus() -> list[dict]:
    corpus = []
    data_dirs = [PROJECT_ROOT / "data" / "standardized", PROJECT_ROOT / "src" / "data" / "standardized"]
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for path in data_dir.rglob("*.md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = split_text(text, chunk_size=1200, overlap=150)
            for index, chunk in enumerate(chunks):
                corpus.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "filename": path.name,
                            "source": str(path.relative_to(PROJECT_ROOT)),
                            "chunk_index": index,
                        },
                    }
                )
    return corpus


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    clean = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def fallback_retrieve(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    use_reranking: bool = True,
) -> list[dict]:
    corpus = load_local_corpus()
    query_terms = tokenize(query)
    scored = []
    for item in corpus:
        doc_terms = tokenize(item["content"])
        if not doc_terms:
            continue
        lexical = len(query_terms & doc_terms) / max(1, len(query_terms))
        density = len(query_terms & doc_terms) / math_sqrt(len(doc_terms))
        score = lexical + 0.15 * density
        if use_reranking:
            score += 0.1 * overlap_score(query, item["content"][:600])
        if score >= score_threshold / 3:
            scored.append({**item, "score": round(score, 4), "source": "fallback_bm25_lite"})
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def math_sqrt(value: int) -> float:
    return value ** 0.5


def run_pipeline_case(
    item: dict,
    config: EvalConfig,
    generate_with_citation: Callable,
    retrieve: Callable,
) -> dict:
    # The generation function does not expose use_reranking, so for fair A/B we
    # retrieve per config and use a compact extractive answer fallback when LLM
    # credentials are absent.
    contexts = retrieve(
        item["question"],
        top_k=config.top_k,
        score_threshold=config.score_threshold,
        use_reranking=config.use_reranking,
    )
    context_chunks = [ctx.get("content", "") for ctx in contexts]

    if os.getenv("OPENAI_API_KEY") and config.use_reranking:
        generated = generate_with_citation(item["question"], top_k=config.top_k)
        answer = generated.get("answer", "")
    else:
        answer = extractive_answer(item["question"], context_chunks)

    return {
        "id": item.get("id"),
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "expected_context": item["expected_context"],
        "answer": answer,
        "retrieved_contexts": context_chunks,
        "sources": [
            {
                "score": ctx.get("score"),
                "source": ctx.get("source") or ctx.get("metadata", {}).get("source"),
                "filename": ctx.get("metadata", {}).get("filename"),
            }
            for ctx in contexts
        ],
    }


def extractive_answer(question: str, contexts: list[str], max_chars: int = 900) -> str:
    if not contexts:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    question_terms = tokenize(question)
    sentences: list[str] = []
    for ctx in contexts:
        sentences.extend(re.split(r"(?<=[.!?。])\s+|\n+", ctx))

    ranked = sorted(
        [s.strip() for s in sentences if s.strip()],
        key=lambda sentence: len(question_terms & tokenize(sentence)),
        reverse=True,
    )
    selected = " ".join(ranked[:3]).strip()
    if not selected:
        selected = contexts[0].strip()
    return selected[:max_chars]


def evaluate_with_heuristics(cases: list[dict]) -> list[dict]:
    evaluated = []
    for case in cases:
        scores = heuristic_metrics(
            question=case["question"],
            answer=case["answer"],
            expected_answer=case["expected_answer"],
            expected_context=case["expected_context"],
            retrieved_contexts=case["retrieved_contexts"],
        )
        evaluated.append({**case, "metrics": scores, "average": round(mean(scores.values()), 4)})
    return evaluated


def evaluate_with_deepeval(cases: list[dict]) -> list[dict]:
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise RuntimeError("DeepEval is not installed. Run: pip install deepeval") from exc

    metrics = {
        "faithfulness": FaithfulnessMetric(threshold=0.7),
        "answer_relevance": AnswerRelevancyMetric(threshold=0.7),
        "context_recall": ContextualRecallMetric(threshold=0.7),
        "context_precision": ContextualPrecisionMetric(threshold=0.7),
    }

    evaluated = []
    for case in cases:
        test_case = LLMTestCase(
            input=case["question"],
            actual_output=case["answer"],
            expected_output=case["expected_answer"],
            retrieval_context=case["retrieved_contexts"],
        )
        scores = {}
        for name, metric in metrics.items():
            metric.measure(test_case)
            scores[name] = round(float(metric.score or 0.0), 4)
        evaluated.append({**case, "metrics": scores, "average": round(mean(scores.values()), 4)})
    return evaluated


def aggregate_scores(evaluated_cases: list[dict]) -> dict[str, float]:
    metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    aggregate = {
        metric: round(mean(case["metrics"][metric] for case in evaluated_cases), 4)
        for metric in metric_names
    }
    aggregate["average"] = round(mean(aggregate.values()), 4)
    return aggregate


def run_ab_evaluation(mode: str) -> dict:
    golden_dataset = load_golden_dataset()
    generate_with_citation, retrieve = import_rag_functions()

    output = {"framework": "DeepEval", "mode": mode, "configs": {}}

    for config in CONFIGS:
        raw_cases = [
            run_pipeline_case(item, config, generate_with_citation, retrieve)
            for item in golden_dataset
        ]
        if mode == "deepeval":
            evaluated_cases = evaluate_with_deepeval(raw_cases)
        else:
            evaluated_cases = evaluate_with_heuristics(raw_cases)

        output["configs"][config.name] = {
            "description": config.description,
            "scores": aggregate_scores(evaluated_cases),
            "cases": evaluated_cases,
        }

    RAW_RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_results(output)
    return output


def worst_performers(cases: list[dict], limit: int = 3) -> list[dict]:
    return sorted(cases, key=lambda case: case["average"])[:limit]


def short(text: str, limit: int = 92) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def infer_failure_stage(case: dict) -> str:
    metrics = case["metrics"]
    lowest = min(metrics, key=metrics.get)
    return {
        "faithfulness": "Generation",
        "answer_relevance": "Generation",
        "context_recall": "Retrieval",
        "context_precision": "Retrieval",
    }[lowest]


def infer_root_cause(case: dict) -> str:
    metrics = case["metrics"]
    if metrics["context_recall"] < 0.45:
        return "Retriever chưa lấy đúng tài liệu hoặc chunk chứa evidence."
    if metrics["context_precision"] < 0.35:
        return "Top-k còn nhiễu, cần rerank hoặc lọc theo metadata."
    if metrics["faithfulness"] < 0.55:
        return "Câu trả lời chưa bám đủ vào context được lấy về."
    return "Cần bổ sung expected_context chi tiết hơn hoặc cải thiện prompt."


def export_results(results: dict) -> None:
    configs = results["configs"]
    first_name, second_name = list(configs.keys())[:2]
    first = configs[first_name]
    second = configs[second_name]

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        "- Framework chọn: DeepEval",
        f"- Chế độ chạy: `{results['mode']}`",
        "- Golden dataset: 15 Q&A pairs",
        f"- Raw output: `{RAW_RESULTS_PATH.name}`",
        "",
        "## Overall Scores",
        "",
        f"| Metric | {first_name} | {second_name} | Delta |",
        "|---|---:|---:|---:|",
    ]

    for metric in ["faithfulness", "answer_relevance", "context_recall", "context_precision", "average"]:
        a = first["scores"][metric]
        b = second["scores"][metric]
        lines.append(f"| {metric} | {a:.3f} | {b:.3f} | {a - b:+.3f} |")

    better = first_name if first["scores"]["average"] >= second["scores"]["average"] else second_name
    lines.extend(
        [
            "",
            "## A/B Comparison Analysis",
            "",
            f"**Config A - {first_name}:** {first['description']}",
            "",
            f"**Config B - {second_name}:** {second['description']}",
            "",
            f"**Kết luận:** `{better}` đang có điểm trung bình tốt hơn. "
            "Nếu config rerank thắng, hệ thống đang hưởng lợi từ bước lọc lại top-k; "
            "nếu config không rerank thắng, cần kiểm tra chất lượng API reranker hoặc ngưỡng score.",
            "",
            "## Worst Performers (Bottom 3)",
            "",
            "| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )

    for index, case in enumerate(worst_performers(first["cases"]), 1):
        m = case["metrics"]
        lines.append(
            f"| {index} | {short(case['question'])} | {m['faithfulness']:.3f} | "
            f"{m['answer_relevance']:.3f} | {m['context_recall']:.3f} | "
            f"{m['context_precision']:.3f} | {infer_failure_stage(case)} | {infer_root_cause(case)} |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "### Cải tiến 1",
            "**Action:** Chuẩn hóa lại dữ liệu crawl, loại bỏ menu/header/footer và sửa lỗi encoding trước khi chunking.",
            "**Expected impact:** Tăng context precision vì top-k bớt nhiễu và chunk chứa nội dung chính rõ hơn.",
            "",
            "### Cải tiến 2",
            "**Action:** Thêm metadata filter theo loại tài liệu (`legal`, `news`) và nguồn (`PLO`, `Kenh14`, `aFamily`) dựa trên intent câu hỏi.",
            "**Expected impact:** Tăng context recall cho câu hỏi pháp luật và câu hỏi tin tức có nguồn cụ thể.",
            "",
            "### Cải tiến 3",
            "**Action:** Tune `top_k`, RRF weight và bật reranker ổn định; log lại source score để debug từng query.",
            "**Expected impact:** Cải thiện cả context recall lẫn faithfulness vì LLM nhận evidence đúng và ít mâu thuẫn hơn.",
            "",
        ]
    )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with DeepEval-compatible metrics.")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "deepeval"],
        default="heuristic",
        help="Use offline heuristic scorer or real DeepEval metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_ab_evaluation(args.mode)
    print(f"Evaluated {len(load_golden_dataset())} cases across {len(result['configs'])} configs.")
    print(f"Markdown report: {RESULTS_PATH}")
    print(f"Raw results: {RAW_RESULTS_PATH}")
