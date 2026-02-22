"""
CI Evaluation Pipeline — MLflow-tracked.

Runs the full RAG pipeline evaluation with experiment tracking:
1. Parse test PDF → collect_all_documents()
2. Chunk + Embed → run_optimized_ingestion_pipeline()
3. Index to Pinecone → create_pinecone_index(namespace='ci_eval')
4. Evaluate retrievers (Vector, BM25, Hybrid)
5. Evaluate RAG quality (Faithfulness, Relevancy)
6. Log everything to MLflow

Usage:
    cd backend && python -m scripts.run_evaluation
"""

import asyncio
import json
import os
import sys
import time

import mlflow

# Ensure backend is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import config
from app.services.llm_service import init_llm, init_embed_model, init_judge_llm, configure_settings
from app.core.document_parser import collect_all_documents
from app.core.chunking import run_optimized_ingestion_pipeline
from app.core.indexing import create_pinecone_index
from app.core.retrieval import create_query_engine
from app.core.evaluation import evaluate_retrievers, evaluate_rag_quality

# Paths
PDF_PATH = os.environ.get("EVAL_PDF_PATH", "../data/attention_is_all_you_need.pdf")
QA_PATH = os.environ.get("EVAL_QA_PATH", "../data/eval/transformer_gold_qa_50.txt")
NAMESPACE = "ci_eval"

# Quality gates
MIN_HIT_RATE = 0.80
MIN_RELEVANCY = 0.85


async def main():
    pipeline_start = time.time()

    # ── MLflow setup ──────────────────────────────────────────────
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("SciChat RAG Pipeline")

    with mlflow.start_run(run_name=f"ci-eval-{int(time.time())}"):
        # Log configuration as params
        mlflow.log_params({
            "llm_model": config.LLM_MODEL,
            "judge_model": config.JUDGE_LLM_MODEL,
            "embed_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "retriever_top_k": config.RETRIEVER_TOP_K,
            "reranker_top_n": config.RERANKER_TOP_N,
            "vector_weight": config.VECTOR_WEIGHT,
            "bm25_weight": config.BM25_WEIGHT,
            "embed_batch_size": config.EMBED_BATCH_SIZE,
            "embed_num_workers": config.EMBED_NUM_WORKERS,
            "pinecone_namespace": NAMESPACE,
            "pdf_path": PDF_PATH,
        })

        # ── Step 1: Init Models ────────────────────────────────────
        print("=" * 60)
        print("SCICHAT RAG EVALUATION PIPELINE (MLflow)")
        print("=" * 60)

        config.validate()
        llm = init_llm()
        judge_llm = init_judge_llm()
        embed_model = init_embed_model()
        configure_settings(llm, embed_model)

        # ── Step 2: Parse PDF ──────────────────────────────────────
        t0 = time.time()
        print(f"\n📄 Step 1: Parsing {PDF_PATH}...")
        docs = collect_all_documents(PDF_PATH)
        parse_time = time.time() - t0

        print(f"   ✅ Parsed {len(docs)} documents in {parse_time:.1f}s")
        mlflow.log_metrics({
            "parse_time_s": round(parse_time, 2),
            "num_documents": len(docs),
        })

        # Log doc type breakdown
        type_counts = {}
        for d in docs:
            t = d.metadata.get("content_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        mlflow.log_params({f"doc_type_{k}": v for k, v in type_counts.items()})
        print(f"   📊 Doc types: {json.dumps(type_counts)}")

        # ── Step 3: Chunk + Embed ──────────────────────────────────
        t0 = time.time()
        print(f"\n🔄 Step 2: Chunking + Embedding...")
        nodes = run_optimized_ingestion_pipeline(docs, embed_model)
        embed_time = time.time() - t0

        print(f"   ✅ Created {len(nodes)} nodes in {embed_time:.1f}s")
        mlflow.log_metrics({
            "embed_time_s": round(embed_time, 2),
            "num_nodes": len(nodes),
        })

        # ── Step 4: Index to Pinecone ──────────────────────────────
        t0 = time.time()
        print(f"\n📌 Step 3: Indexing to Pinecone (namespace={NAMESPACE})...")
        index = create_pinecone_index(nodes, embed_model, namespace=NAMESPACE)
        index_time = time.time() - t0

        print(f"   ✅ Indexed in {index_time:.1f}s")
        mlflow.log_metric("index_time_s", round(index_time, 2))

        # ── Step 5: Retriever Evaluation ───────────────────────────
        t0 = time.time()
        print(f"\n📊 Step 4: Retriever evaluation...")
        retriever_metrics = await evaluate_retrievers(index, nodes, llm)
        retriever_time = time.time() - t0

        mlflow.log_metric("retriever_eval_time_s", round(retriever_time, 2))

        for m in retriever_metrics:
            prefix = f"retriever_{m.retriever_name}"
            mlflow.log_metrics({
                f"{prefix}_mrr": round(m.mrr, 4),
                f"{prefix}_hit_rate": round(m.hit_rate, 4),
                f"{prefix}_precision": round(m.precision, 4),
                f"{prefix}_recall": round(m.recall, 4),
            })
            print(f"   {m.retriever_name}: MRR={m.mrr:.3f} HitRate={m.hit_rate:.3f} "
                  f"Precision={m.precision:.3f} Recall={m.recall:.3f}")

        # ── Step 6: RAG Quality Evaluation ─────────────────────────
        t0 = time.time()
        print(f"\n🎯 Step 5: RAG quality evaluation...")
        engine = create_query_engine(index, nodes, llm)
        rag_metrics = await evaluate_rag_quality(engine, judge_llm, QA_PATH)
        rag_time = time.time() - t0

        mlflow.log_metrics({
            "rag_eval_time_s": round(rag_time, 2),
            "avg_faithfulness": round(rag_metrics.avg_faithfulness, 4),
            "avg_relevancy": round(rag_metrics.avg_relevancy, 4),
            "num_questions_evaluated": rag_metrics.num_questions,
        })
        print(f"   Faithfulness: {rag_metrics.avg_faithfulness:.3f}")
        print(f"   Relevancy:    {rag_metrics.avg_relevancy:.3f}")
        print(f"   Questions:    {rag_metrics.num_questions}")

        # ── Total pipeline time ────────────────────────────────────
        total_time = time.time() - pipeline_start
        mlflow.log_metric("total_pipeline_time_s", round(total_time, 2))

        # ── Results summary ────────────────────────────────────────
        results = {
            "retriever_metrics": [
                {"name": m.retriever_name, "mrr": m.mrr, "hit_rate": m.hit_rate,
                 "precision": m.precision, "recall": m.recall}
                for m in retriever_metrics
            ],
            "rag_quality": {
                "faithfulness": rag_metrics.avg_faithfulness,
                "relevancy": rag_metrics.avg_relevancy,
            },
            "timing": {
                "parse_s": round(parse_time, 2),
                "embed_s": round(embed_time, 2),
                "index_s": round(index_time, 2),
                "retriever_eval_s": round(retriever_time, 2),
                "rag_eval_s": round(rag_time, 2),
                "total_s": round(total_time, 2),
            },
        }
        print(f"\n{'=' * 60}")
        print("RESULTS:")
        print(json.dumps(results, indent=2))

        # Save results as artifact
        results_path = "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(results_path)

        # ── Quality Gates ──────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print("QUALITY GATES:")
        gate_passed = True

        hybrid = next((m for m in retriever_metrics if m.retriever_name == "hybrid"), None)
        if hybrid:
            if hybrid.hit_rate < MIN_HIT_RATE:
                print(f"   ❌ Hybrid Hit Rate {hybrid.hit_rate:.3f} < {MIN_HIT_RATE}")
                gate_passed = False
            else:
                print(f"   ✅ Hybrid Hit Rate {hybrid.hit_rate:.3f} >= {MIN_HIT_RATE}")

        if rag_metrics.avg_relevancy < MIN_RELEVANCY:
            print(f"   ❌ Relevancy {rag_metrics.avg_relevancy:.3f} < {MIN_RELEVANCY}")
            gate_passed = False
        else:
            print(f"   ✅ Relevancy {rag_metrics.avg_relevancy:.3f} >= {MIN_RELEVANCY}")

        mlflow.log_metric("quality_gate_passed", 1.0 if gate_passed else 0.0)

        if not gate_passed:
            print("\n❌ QUALITY GATES FAILED")
            sys.exit(1)

        print("\n✅ All quality gates passed!")


if __name__ == "__main__":
    asyncio.run(main())
