"""
evaluation/evaluator.py — RAGAS evaluation pipeline.
Measures 3 core production RAG metrics:
  - Faithfulness: Does the answer stick to the context? (no hallucination)
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are the retrieved chunks actually useful?
"""
from typing import List, Dict, Any
from loguru import logger


def evaluate_rag(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str] = None
) -> Dict[str, float]:
    """
    Run RAGAS evaluation on a set of Q&A pairs.

    Args:
        questions: List of user questions.
        answers: List of generated answers.
        contexts: List of context lists (one list of chunks per question).
        ground_truths: Optional reference answers for additional metrics.

    Returns:
        Dict of metric names to scores (0.0 - 1.0).
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
    except ImportError:
        logger.error("ragas not installed. Run: pip install ragas datasets")
        return {}

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_precision]

    logger.info(f"Running RAGAS evaluation on {len(questions)} samples...")
    try:
        result = evaluate(dataset, metrics=metrics)
        scores = {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevancy": round(float(result["answer_relevancy"]), 4),
            "context_precision": round(float(result["context_precision"]), 4),
        }
        logger.info(f"RAGAS scores: {scores}")
        return scores
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"error": str(e)}


def run_quick_eval(qa_pairs: List[Dict]) -> Dict[str, float]:
    """
    Convenience wrapper: run a quick eval from the full RAG pipeline.

    Args:
        qa_pairs: List of {"question": ..., "ground_truth": ...} dicts.

    Returns:
        RAGAS score dict.
    """
    from src.retrieval.retriever import retrieve
    from src.retrieval.reranker import rerank
    from src.generation.generator import generate_response

    questions, answers, contexts, ground_truths = [], [], [], []

    for pair in qa_pairs:
        q = pair["question"]
        gt = pair.get("ground_truth", "")

        candidates = retrieve(q)
        reranked = rerank(q, candidates)
        result = generate_response(q, reranked)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["content"] for c in reranked])
        ground_truths.append(gt)

        logger.debug(f"Eval Q: {q[:60]} | Answer: {result['answer'][:60]}")

    return evaluate_rag(questions, answers, contexts, ground_truths or None)
