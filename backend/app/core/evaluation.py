"""
Evaluation pipeline — extracted from notebook Cells 13-15.

Pipeline evaluation only — runs on code changes, NOT per-upload.

Two evaluation modes:
1. Retriever metrics: Generate QA pairs from nodes → MRR, Hit Rate, Precision, Recall
2. RAG quality: Gold QA dataset → Faithfulness, Relevancy (LLM judge)
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    RetrieverEvaluator,
    generate_question_context_pairs,
)
from llama_index.core.query_engine import RetrieverQueryEngine

from app.api.schemas import RagQualityMetrics, RetrieverMetrics
from app.config import config
from app.core.retrieval import create_all_retrievers

logger = logging.getLogger(__name__)


# =============================================================================
# RETRIEVER EVALUATION (from notebook Cell 13-14)
# =============================================================================

async def evaluate_retrievers(
    index: VectorStoreIndex,
    nodes: List,
    llm,
) -> List[RetrieverMetrics]:
    """
    Evaluate all retrievers (Vector, BM25, Hybrid) on generated QA pairs.

    Steps:
    1. Generate question-context pairs from current nodes
    2. Evaluate each retriever against generated pairs
    3. Return MRR, Hit Rate, Precision, Recall for each

    From notebook Cells 13-14 — adapted for async.
    """
    logger.info("📊 Generating QA pairs from nodes...")

    # Step 1: Generate QA dataset from nodes
    qa_dataset = generate_question_context_pairs(
        nodes=nodes,
        llm=llm,
        num_questions_per_chunk=1,
    )

    logger.info(f"   ✅ Generated {len(qa_dataset.queries)} question-context pairs")

    # Step 2: Create all retrievers
    retrievers = create_all_retrievers(index, nodes)

    results = []

    for name, retriever in retrievers.items():
        logger.info(f"   Evaluating {name} retriever...")

        retriever_eval = RetrieverEvaluator.from_metric_names(
            config.EVAL_METRICS_RETRIEVER,
            retriever=retriever,
        )

        eval_results = await retriever_eval.aevaluate_dataset(qa_dataset)

        # Compute averages
        metrics = {metric: 0.0 for metric in config.EVAL_METRICS_RETRIEVER}
        for result in eval_results:
            for metric_name in config.EVAL_METRICS_RETRIEVER:
                metric_dict = result.metric_vals_dict
                if metric_name in metric_dict:
                    metrics[metric_name] += metric_dict[metric_name]

        n = len(eval_results) if eval_results else 1
        for metric_name in metrics:
            metrics[metric_name] /= n

        results.append(RetrieverMetrics(
            retriever_name=name,
            mrr=metrics.get("mrr", 0.0),
            hit_rate=metrics.get("hit_rate", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
        ))

        logger.info(
            f"   {name}: MRR={metrics.get('mrr', 0):.3f}, "
            f"HitRate={metrics.get('hit_rate', 0):.3f}, "
            f"Precision={metrics.get('precision', 0):.3f}, "
            f"Recall={metrics.get('recall', 0):.3f}"
        )

    return results


# =============================================================================
# RAG QUALITY EVALUATION (from notebook Cell 15)
# =============================================================================

def load_gold_qa_questions(qa_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load gold QA dataset from file.
    Supports two formats:
    - Tab-separated: question\\tanswer (one pair per line)
    - Question-only: one question per line (answer left empty)
    """
    path = qa_path or config.GOLD_QA_PATH
    if not os.path.exists(path):
        logger.warning(f"⚠️ Gold QA file not found: {path}")
        return []

    questions = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                questions.append({
                    "question": parts[0].strip(),
                    "answer": parts[1].strip(),
                })
            else:
                # Question-only format
                questions.append({
                    "question": line,
                    "answer": "",
                })

    logger.info(f"📋 Loaded {len(questions)} gold QA pairs from {path}")
    return questions


async def evaluate_rag_quality(
    query_engine: RetrieverQueryEngine,
    judge_llm,
    qa_path: Optional[str] = None,
) -> RagQualityMetrics:
    """
    Evaluate RAG quality using gold QA dataset.
    Measures Faithfulness and Relevancy using LLM judge.

    From notebook Cell 15 — adapted for async.
    """
    questions = load_gold_qa_questions(qa_path)

    if not questions:
        return RagQualityMetrics()

    faithfulness_eval = FaithfulnessEvaluator(llm=judge_llm)
    relevancy_eval = RelevancyEvaluator(llm=judge_llm)

    total_faithfulness = 0.0
    total_relevancy = 0.0
    evaluated = 0

    for i, qa in enumerate(questions):
        try:
            logger.info(f"   Evaluating Q{i+1}/{len(questions)}: {qa['question'][:60]}...")

            # Get response from RAG
            response = await query_engine.aquery(qa["question"])

            # Evaluate faithfulness
            faith_result = await faithfulness_eval.aevaluate_response(response=response)
            faith_score = faith_result.score if faith_result.score is not None else 0.0

            # Evaluate relevancy
            rel_result = await relevancy_eval.aevaluate_response(
                query=qa["question"], response=response
            )
            rel_score = rel_result.score if rel_result.score is not None else 0.0

            total_faithfulness += faith_score
            total_relevancy += rel_score
            evaluated += 1

            logger.info(f"   Q{i+1}: Faithfulness={faith_score:.2f}, Relevancy={rel_score:.2f}")

        except Exception as e:
            logger.error(f"   ❌ Error evaluating Q{i+1}: {e}")

    avg_faithfulness = total_faithfulness / evaluated if evaluated > 0 else 0.0
    avg_relevancy = total_relevancy / evaluated if evaluated > 0 else 0.0

    logger.info(f"\n📊 RAG QUALITY SUMMARY:")
    logger.info(f"   Avg Faithfulness: {avg_faithfulness:.3f}")
    logger.info(f"   Avg Relevancy: {avg_relevancy:.3f}")
    logger.info(f"   Questions evaluated: {evaluated}/{len(questions)}")

    return RagQualityMetrics(
        avg_faithfulness=avg_faithfulness,
        avg_relevancy=avg_relevancy,
        num_questions=evaluated,
    )
