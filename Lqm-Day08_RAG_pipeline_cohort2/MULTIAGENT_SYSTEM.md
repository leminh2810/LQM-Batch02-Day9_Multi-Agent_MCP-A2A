# Multi-Agent RAG System

The Streamlit app now prefers `src.multiagent_rag.run_multiagent_rag` as its
main backend. The system keeps the original Task 9 retrieval pipeline and Task
10 prompt utilities, then wraps them with small agents:

1. `QueryPlannerAgent`: classifies the query as legal, news, mixed, or general
   and chooses retrieval settings.
2. `RetrievalAgent`: calls the hybrid retrieval pipeline with reranking and
   PageIndex fallback.
3. `EvidenceCuratorAgent`: removes duplicate chunks and reorders context to
   reduce lost-in-the-middle effects.
4. `AnswerWriterAgent`: generates a cited answer with OpenAI when
   `OPENAI_API_KEY` is available, otherwise returns an extractive cited answer.
5. `VerificationAgent`: checks whether the final answer has context-backed
   citations or explicitly says the evidence is insufficient.

The backend returns the same app contract as the old generator:

```python
{
    "answer": "...",
    "sources": [...],
    "retrieval_source": "hybrid | pageindex | none",
    "agent_trace": [...],
}
```

The app renders `agent_trace` in an expander so demos can show what each agent
did for a user question.

