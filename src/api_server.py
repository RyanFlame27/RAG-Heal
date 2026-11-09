"""Serve the final pipeline and dashboard: python -m src.api_server."""
import json
import logging
import math
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)
pipeline_instance = classifier_instance = None
_query_lock = threading.Lock()
FRONTEND = config.PROJECT_ROOT / "frontend"


def thresholds():
    return {"retrieval_similarity": config.RETRIEVAL_SIM_THRESHOLD,
            "entailment_score": config.ENTAILMENT_THRESHOLD,
            "mean_entropy": config.ENTROPY_THRESHOLD,
            "coverage_score": config.COVERAGE_SIM_THRESHOLD}


def json_safe(value):
    """Keep missing/invalid measurements null throughout nested traces."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def set_pipeline(pipeline, classifier=None):
    """Reuse the notebook's models instead of loading them twice."""
    global pipeline_instance, classifier_instance
    pipeline_instance, classifier_instance = pipeline, classifier


def initialize_runtime():
    if pipeline_instance is not None:
        return
    from src.prepare_deployment import check_assets
    check_assets(require_index=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Connect a CUDA GPU (T4 or equivalent) before loading Mistral.")
    from src.classifier.predict import load_classifier
    from src.pipeline.rag import RAGPipeline
    from src.detection.nli_entailment import _load_nli_model
    classifier = load_classifier()
    pipeline = RAGPipeline(recovery_version=2)
    if pipeline.index.ntotal != 366742 or len(pipeline.chunks) != 366742 or pipeline.index.d != 384:
        raise ValueError("Loaded FAISS index differs from the final 366,742-vector corpus.")
    _load_nli_model()
    pipeline.evidence_retriever._load()
    set_pipeline(pipeline, classifier)


def get_runtime():
    if pipeline_instance is None:
        raise HTTPException(503, "Pipeline is not ready. Run the notebook model-loading cell first.")
    from src.classifier.predict import load_classifier
    from src.recovery.recovery_matrix import RecoveryMatrix
    return pipeline_instance, classifier_instance or load_classifier(), RecoveryMatrix(pipeline_instance)


@asynccontextmanager
async def lifespan(app):
    if os.getenv("RAG_AUTO_INIT", "1") == "1" and pipeline_instance is None:
        await run_in_threadpool(initialize_runtime)
    yield


app = FastAPI(title="Self-Healing RAG API", version=config.RELEASE_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("RAG_ALLOWED_ORIGINS", "*").split(",") if x.strip()],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class DemoRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=80)


@app.get("/health")
def health_check():
    try:
        import torch
        gpu = bool(torch.cuda.is_available())
    except ImportError:
        gpu = False
    ready = pipeline_instance is not None
    payload = {"status": "ok" if ready else "not_ready", "ready": ready,
               "gpu_available": gpu, "version": config.RELEASE_VERSION,
               "classifier_backend": config.FAILURE_CLASSIFIER_BACKEND,
               "index_vectors": int(pipeline_instance.index.ntotal) if ready else None}
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/metadata")
def metadata():
    return {"version": config.RELEASE_VERSION, "classifier_backend": "bert",
            "classifier_checkpoint": config.BERT_CLASSIFIER_DIR.name,
            "generator": config.LLM_MODEL_NAME, "recovery_version": config.RECOVERY_VERSION,
            "thresholds": thresholds(), "external_web_search": False, "demo_fault_injection": True,
            "recovery": {"candidates_per_search": 24, "selected_passages": 3,
                         "adjacent_chunk_radius": 1, "max_new_tokens": 64,
                         "query_rewrite": "retrieval_failure only"}}


@app.get("/benchmark")
def benchmark():
    """Recorded results, kept separate from live inference."""
    path = config.BENCHMARK_DIR / "release_summary.json"
    if not path.is_file():
        raise HTTPException(404, "Recorded benchmark summary is not included.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/query")
def process_query_endpoint(req: QueryRequest) -> dict[str, Any]:
    query = req.query.strip()
    if not query:
        raise HTTPException(422, "Enter a non-empty question.")
    return run_request(query)


@app.get("/demo/cases")
def demo_catalog():
    from src.demo_cases import demo_cases
    return {"mode": "injected_demo", "cases": demo_cases(),
            "note": "Left: supplied faulty state. Right: live BERT-routed recovery, not a saved correct answer. This is a demonstration, not a new benchmark."}


@app.post("/demo/run")
def process_demo_endpoint(req: DemoRequest) -> dict[str, Any]:
    from src.demo_cases import get_demo_case
    case = get_demo_case(req.case_id)
    if case is None:
        raise HTTPException(404, "Unknown demo case. Choose one of the listed cases.")
    return run_request(case["query"], demo_case=case)


def run_request(query, demo_case=None):
    if not _query_lock.acquire(blocking=False):
        raise HTTPException(429, "The GPU is answering another question. Please try again shortly.")
    started = time.perf_counter()
    try:
        pipeline, classifier, matrix = get_runtime()
        if demo_case is None:
            initial = pipeline.run_rag_with_detection(query)
        else:
            from src.demo_cases import injected_initial
            initial = injected_initial(demo_case, matrix)
        signals = initial["signals"]
        prediction = classifier.predict(signals, query=query,
            contexts=initial["context_texts"], answer=initial["answer"])
        recovered = matrix.heal(query, initial, prediction["predicted_label"])
        assessment = recovered["assessment"]
        entailment = signals.get("entailment_score")
        grounding_display = ("N/A (Abstention)" if signals.get("is_refusal") else
            "N/A (measurement unavailable)" if entailment is None or not math.isfinite(entailment) else
            f"{entailment:.4f} " + ("PASS" if entailment >= config.ENTAILMENT_THRESHOLD else "FAIL"))
        result = {
            "execution_mode": "injected_demo" if demo_case else "live_query",
            "query": query, "initial_answer": initial["answer"],
            "retrieval_sim": signals.get("retrieval_similarity"),
            "coverage_score": signals.get("coverage_score"),
            "grounding_score": entailment, "grounding_score_display": grounding_display,
            "token_entropy": signals.get("mean_entropy"), "entropy_threshold": config.ENTROPY_THRESHOLD,
            "predicted_failure": prediction["predicted_label"].upper(),
            "classifier_backend": prediction.get("classifier_backend", "bert"),
            "class_probabilities": prediction.get("probabilities", {}),
            "action_taken": recovered["action_taken"], "healed_answer": recovered["healed_answer"],
            "verified": recovered["verified"], "grounded_answer": assessment["grounded_answer"],
            "response_outcome": assessment["outcome"], "assessment": assessment,
            "pre_signals": signals, "post_signals": recovered["post_signals"],
            "initial_contexts": initial["context_texts"],
            "final_contexts": recovered["post_context_texts"],
            "retrieved_chunks": initial.get("retrieved_chunks", []),
            "recovery_trace": recovered.get("recovery_trace", {}),
            "verification_status": recovered["verification_notes"],
            "thresholds": thresholds(), "version": config.RELEASE_VERSION,
            "elapsed_seconds": round(time.perf_counter()-started, 3),
        }
        if demo_case:
            result["demo"] = dict(demo_case,
                initial_answer_source="Supplied faulty answer; not a live Mistral generation",
                final_answer_source="Live recovery pipeline; no reference answer supplied",
                initial_entropy_note="Not measured: the initial answer was injected, not generated",
                benchmark_result=False)
        return json_safe(result)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(500, "Inference failed. Check the Colab runtime log and retry.") from None
    finally:
        _query_lock.release()


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
