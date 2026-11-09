"""
Interactive Gradio Web Dashboard for Self-Healing RAG Pipeline.

Provides a modern web UI showcasing:
1. Live Query Input with Pre-set Demo Presets (Clean, Hallucination, Coverage, Retrieval)
2. Live Signal Radar/Bar Metrics (Retrieval, Grounding, Entropy, Coverage)
3. Classifier Decision & Active Recovery Route
4. Pre vs Post Healing Comparison with Closed-Loop Quality Verification
"""

import math
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

try:
    import gradio as gr
except ImportError:
    gr = None


def create_demo_app(pipeline, classifier, recovery_matrix):
    """Create and return a configured Gradio Blocks web application."""
    if gr is None:
        raise ImportError("Gradio is not installed. Please run: pip install gradio")

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    )

    def process_query(query_text: str):
        if not query_text or not query_text.strip():
            return (
                "Please enter a question to query the pipeline.",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            )

        # 1. Execute initial RAG with signal audit
        initial_res = pipeline.run_rag_with_detection(query_text)
        signals = initial_res.get("signals", {})
        ret_sim = signals.get("retrieval_similarity", 0.0)
        entail = signals.get("entailment_score", 0.0)
        tok_ent = signals.get("mean_entropy")
        kb_cov = signals.get("coverage_score", 0.0)

        # 2. Predict failure type
        pred = classifier.predict(signals, query=query_text,
                                  contexts=initial_res["context_texts"], answer=initial_res["answer"])
        pred_label = pred["predicted_label"].upper()

        # 3. Closed-Loop Recovery
        healed_data = recovery_matrix.heal(query_text, initial_res, pred["predicted_label"])
        healed_answer = healed_data.get("healed_answer", initial_res["answer"])
        action_taken = healed_data.get("action_taken", "None")
        verified = healed_data.get("verified", False)
        post_signals = healed_data.get("post_signals", signals)

        # Status badges
        ret_badge = f"{ret_sim:.4f}  " + ("🟢 PASS" if ret_sim >= config.RETRIEVAL_SIM_THRESHOLD else "🔴 FAIL")
        ent_badge = "N/A (abstention)" if signals.get("is_refusal") else f"{entail:.4f}  " + ("🟢 PASS" if entail >= config.ENTAILMENT_THRESHOLD else "🔴 FAIL")
        tok_badge = "N/A (measurement unavailable)" if tok_ent is None else f"{tok_ent:.4f}  " + ("🟢 PASS" if tok_ent < config.ENTROPY_THRESHOLD else "🔴 FAIL")
        cov_badge = f"{kb_cov:.4f}  " + ("🟢 PASS" if kb_cov >= config.COVERAGE_SIM_THRESHOLD else "🔴 FAIL")

        v_status = healed_data["verification_notes"]

        return (
            initial_res["answer"],
            ret_badge,
            ent_badge,
            tok_badge,
            cov_badge,
            pred_label,
            action_taken,
            healed_answer,
            v_status,
        )

    with gr.Blocks(title="Self-Healing RAG Control Center", theme=custom_theme) as demo:
        gr.Markdown(
            """
            # 🛡️ Self-Healing RAG Control Center
            ### Real-Time Detection, Multi-Class Failure Classification & Closed-Loop Autonomous Recovery
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                query_input = gr.Textbox(
                    label="Enter Question / Prompt",
                    placeholder="Ask a question or select one of the presets below...",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("🚀 Run Self-Healing RAG", variant="primary")
                    clear_btn = gr.Button("Clear")

                gr.Markdown("#### 💡 Quick Test Presets (Click to Load):")
                with gr.Row():
                    p1 = gr.Button("🟢 In-Domain (Clean)", size="sm")
                    p2 = gr.Button("🔴 False Claim (Grounding)", size="sm")
                    p3 = gr.Button("🔴 Out-of-Domain (Coverage)", size="sm")
                    p4 = gr.Button("🔴 Gibberish (Retrieval)", size="sm")

            with gr.Column(scale=4):
                gr.Markdown("### 📊 Real-Time Signal Audits")
                ret_out = gr.Textbox(label="1. Retrieval Similarity (Threshold: ≥ 0.45)")
                ent_out = gr.Textbox(label="2. Grounding Compatibility (Threshold: ≥ 0.50)")
                tok_out = gr.Textbox(label=f"3. Token Entropy in nats (Threshold: < {config.ENTROPY_THRESHOLD:.2f})")
                cov_out = gr.Textbox(label="4. Knowledge Base Coverage (Threshold: ≥ 0.45)")

        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚠️ Phase 1: Initial Answer (Pre-Healing)")
                init_ans_out = gr.Textbox(label="Initial LLM Output", lines=5)
                pred_label_out = gr.Label(label="Failure Classifier Prediction")

            with gr.Column():
                gr.Markdown("### 🛠️ Phase 2: Autonomous Recovery & Re-Verification")
                action_out = gr.Textbox(label="Triggered Recovery Action", lines=2)
                healed_ans_out = gr.Textbox(label="Final Healed Output (Delivered to User)", lines=5)
                verify_out = gr.Textbox(label="Closed-Loop Verification Status")

        # Preset bindings
        p1.click(lambda: "What is photosynthesis and how do plants convert sunlight into energy?", None, query_input)
        p2.click(lambda: "Why did the Eiffel Tower collapse in 2019?", None, query_input)
        p3.click(lambda: "What is the wifi password for Starbucks in Mumbai airport?", None, query_input)
        p4.click(lambda: "qwzxy7890 vbnm 9988 zxcvbnm 123456789 qwertyuiop zzzzzzzz", None, query_input)

        # Trigger execution
        submit_btn.click(
            process_query,
            inputs=[query_input],
            outputs=[
                init_ans_out,
                ret_out,
                ent_out,
                tok_out,
                cov_out,
                pred_label_out,
                action_out,
                healed_ans_out,
                verify_out,
            ],
        )
        clear_btn.click(lambda: [""] * 9, None, [
            query_input, init_ans_out, ret_out, ent_out, tok_out, cov_out,
            pred_label_out, action_out, healed_ans_out, verify_out
        ])

    return demo


if __name__ == "__main__":
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import load_index
    from src.generation.llm import LLMWrapper
    from src.pipeline.rag import RAGPipeline
    from src.classifier.predict import FailureClassifier
    from src.recovery.recovery_matrix import RecoveryMatrix

    print("Launching Self-Healing RAG UI...")
    emb = Embedder()
    idx, chk = load_index()
    llm = LLMWrapper()
    pipe = RAGPipeline(embedder=emb, llm=llm, index=idx, chunks=chk)
    clf = FailureClassifier()
    mat = RecoveryMatrix(pipe)

    app = create_demo_app(pipe, clf, mat)
    app.launch(share=True)
