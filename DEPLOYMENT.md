# Deployment handoff — 24 October 2026

## Use the original Colab workflow

1. Clone the repository with Git LFS using the [README instructions](README.md#clone-the-project), or extract the `pjt/` folder from the original `Self_Healing_RAG_Updated_Project.zip`.
2. Upload the project files to **Google Drive → MyDrive → pjt**. Check that `pjt/data/bert_failure_classifier_runtime_v4/model.safetensors` uploaded completely (437,967,876 bytes). If replacing an existing project, merge the new source and checkpoint files while retaining existing `data/faiss_index/` and `data/faiss_index_with_source_passages/` caches.
3. Upload/open `pjt/notebooks/05_final_pipeline_demo.ipynb` in Colab. Select **Runtime → Change runtime type → T4 GPU** and start a fresh runtime.
4. Run **Step 1** to mount Drive, copy the project into `/content/pjt`, install the recorded dependency versions and check the GPU.
5. Run **Step 2** to check/rebuild the search index. Missing files are built from the same Wikipedia snapshot and the included source paragraphs. Expected sizes are 366,568 base chunks and 366,742 final vectors. A mismatched existing corpus stops with a clear error instead of silently using it. The indices are saved back to Drive; repeat sessions reuse them.
6. Run **Step 3** to load the accepted BERT, Mistral-7B NF4 generator, MiniLM embedder, DeBERTa NLI model and passage reranker. This does not train a new classifier.
7. Run **Step 4** to start the FastAPI server. It must print `ready: True` and runtime version `4.2`.
8. Run **Step 5** and open the printed dashboard link. That URL serves both the HTML page and the API. Keep the Colab runtime connected.

Cloudflare quick tunnels provide a temporary URL that changes when restarted. The notebook retains the original temporary-tunnel approach; it does not configure a permanent production domain. See the [Cloudflare quick-tunnel documentation](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

## Separately hosting the frontend

Copy the complete `frontend/` folder, including `static/`, to your existing static host. There is no npm build step. Open **Gateway** and paste the backend tunnel URL printed by Colab, without `/query` at the end. The frontend calls `/health`, `/metadata` and `/query` on that URL. Stored evaluation cards load from the included `static/benchmark-data.js`, so they stay clearly separate from live results.

If moving from an old saved gateway, click **Use this server** when using the notebook's dashboard link. This replaces the stale saved endpoint with the page's current origin.

## What to check before the demo

- Backend label says **Backend ready**.
- The runtime health response reports version `4.2` and `index_vectors: 366742`.
- Run a question and inspect the BERT route, initial/final answer, before/after signal table, evidence and recovery trace.
- A refusal must display **Refusal**, and an answer failing checks must display **Unverified**.
- The main page runs ordinary live questions only. Start with the **College president** question preset, then click **Audit & Self-Heal**. See `LIVE_DEMO.md` for the actual T4 checks.
- Saved dataset examples are not promised live answers. Clicking **Run live question** uses fresh retrieval without injecting the saved corrupted context.

## Troubleshooting

| Message | Action |
|---|---|
| Checkpoint not found or wrong hash | Re-upload the supplied `bert_failure_classifier_runtime_v4` directory; do not retrain as a workaround. |
| Missing deployment assets: index files | Run Step 2 once on the T4; it builds/reuses and persists the index. |
| Corpus size differs from evaluated corpus | Check the supplied splitter/dataset versions and old cache. Keep a backup of an incompatible cache before moving it aside and rebuilding. |
| GPU unavailable | Reconnect a T4 runtime, then rerun the notebook in order. |
| Hugging Face download fails | Check runtime internet access and any Hugging Face access requirements shown by the download error, then rerun the loading cell. |
| Backend unavailable / tunnel changed | Verify Step 4 is ready, restart Step 5, and update Gateway if using a separate frontend. |
| 429: GPU answering another question | Wait for the current query to finish, then retry. |
| Request timed out | Inspect the Colab/server output; keep one active query and retry after the runtime is ready. |

The full model stack needs a working CUDA runtime. This handoff was checked locally with API fixtures and the preserved final T4 evaluation records; a fresh Colab deployment still needs to be started on your teammate's account.
