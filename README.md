# Self-Healing RAG: BERT-Guided Failure Detection and Recovery

RAG Heal detects retrieval and generation failures, then retrieves fresh evidence to repair or decline unreliable answers. Runtime 4.2 runs on **Google Drive → Colab T4 → Cloudflare tunnel → HTML dashboard**. See [DEPLOYMENT.md](DEPLOYMENT.md).

## Clone the project

With Git LFS installed, clone into `pjt` to match the Colab notebook's expected folder name. The final BERT checkpoint is stored in Git LFS.

```bash
git lfs install
git clone https://github.com/RyanFlame27/RAG-Heal.git pjt
cd pjt
git lfs pull
```

## Quick start on Colab

1. Copy the project files from **pjt** into `MyDrive/pjt` in Google Drive, including `data/bert_failure_classifier_runtime_v4/`. Upload only the working files. If you are using the original transfer ZIP, extract its **pjt** folder first.
2. If you already have the old project's FAISS index in that Drive folder, keep it when merging the new files. The setup checks its size before reuse.
3. Open `notebooks/05_final_pipeline_demo.ipynb` in Colab and select a **T4 GPU**. Use a fresh runtime.
4. Run the five steps in order. The notebook installs dependencies, prepares/reuses the index, loads the models, starts the API, and prints a dashboard link.
5. Open that link. Frontend and API use the same server automatically. If you host `frontend/` separately, paste the printed backend URL into its Gateway settings instead.

No BERT training or benchmark rerun is needed for deployment. The first launch uses the included builder and source paragraphs to build missing indices on the T4 and saves them in Drive for reuse. Pretrained Mistral, MiniLM, NLI and reranker weights download from Hugging Face on first use.

## Final implementation included

- Runtime-adapted, five-class BERT routes the **question + context + initial answer**. Its accepted weights are in `data/bert_failure_classifier_runtime_v4/`. It is separate from the DeBERTa NLI audit model.
- Local FAISS corpus: 366,568 Wikipedia chunks, extended by 174 chunks from 40 original source paragraphs selected using training questions. The final index has **366,742 vectors**.
- Initial Mistral-7B generation uses 4-bit NF4, greedy decoding and a maximum of 64 new tokens.
- Every failure route performs fresh local search, restores adjacent context, removes duplicates, reranks passages and selects up to three evidence passages for regeneration. Retrieval failure additionally rewrites the query.
- Recovery can format a single unambiguous requested year or percentage only when the literal value is present in evidence. It retains qualifiers and raw generation output.
- Response selection preserves an already supported initial answer if a retry weakens it; otherwise it selects an eligible answer or refuses. The final status remains **supported**, **refusal** or **unverified** according to the shared audit.
- No external web search is used for answers. Python implements routing/recovery; LangChain's text splitter is used only when preparing the corpus.

The deployed UI displays BERT probabilities, before/after signals, the selected evidence, the recovery trace and a downloadable JSON query record. Probabilities are routing probabilities, not factual-accuracy scores. The old fabricated demo outputs have been replaced by saved evaluation examples. The main performance comparison is the recorded Self-RAG adaptation; local CRAG is not presented as a full baseline.

The main console runs ordinary live questions through `/query`. Both boxes show
actual generation and recovery results. Its example buttons fill the question;
they do not supply faulty inputs or preset answers. See [LIVE_DEMO.md](LIVE_DEMO.md)
for examples checked on the current T4 session.

## Recorded results

The 1,000-row constructed dataset uses 704 training, 158 validation and 138 regression rows. The final run completed those 138 rows with no execution errors. These are reused regression IDs, not a new blind test.

| Measure | Count | Result |
|---|---:|---:|
| Clean-case exact-answer accuracy | 27/30 | 90.00% |
| Exact correctness among answered factual QA | 60/63 | 95.24% |
| Expected refusals on designated missing-information challenges | 18/18 | 100.00% |
| Answer rate across reference-scored QA | 63/120 | 52.50% |
| Exact match across all reference-scored QA | 60/120 | 50.00% |
| Token F1 across reference-scored QA | 120 QA cases | 51.63% |

The 57 factual-QA refusals are recorded separately; they are not all independently established as appropriate refusals. A short extractive answer may remain unverified under the NLI/signal audit. Three nonmatching answers preserved the expected fact with extra words, but no broad semantic-accuracy score was computed.

The recorded Self-RAG adaptation had 7.50% exact match and 15.89% token F1 on the same 120 QA IDs. Its generator, corpus and answer format differed, so the gap does not isolate architecture quality. The UI identifies the configurations and this limitation.

## Project layout

```text
pjt/
  config.py
  DEPLOYMENT.md
  requirements.txt
  notebooks/01_build_index.ipynb
  notebooks/05_final_pipeline_demo.ipynb
  frontend/index.html
  frontend/static/                  # JS, saved benchmark data, pinned icons
  src/api_server.py                 # /, /query, /health, /metadata, /benchmark
  src/prepare_deployment.py         # check assets and build/reuse corpus
  src/classifier/                   # accepted BERT loader + research training code
  src/retrieval/, generation/, detection/, recovery/, pipeline/
  data/bert_failure_classifier_runtime_v4/
  data/source_documents.json
  benchmark/failure_dataset.json
  benchmark/release_summary.json
  benchmark/final_results/          # saved per-case records and provenance
  tests/
  RELEASE_MANIFEST.json
```

The HTML dashboard runs directly from `frontend/` without a build step. `RELEASE_MANIFEST.json` records checksums for the original transfer archive. The repository also contains subsequent live-demo updates.

## Direct launch on a CUDA machine

Run from this unpacked project root with Python 3.10 or newer. Colab supplies the CUDA PyTorch build; on another machine install a CUDA build appropriate to the host before installing the requirements.

```bash
python -m pip install -r requirements.txt
python -m src.prepare_deployment
python -m src.api_server
```

Open `http://localhost:8000`. Use one API worker so that GPU models are not loaded multiple times. The service returns 503 until ready and 429 if another GPU query is already running. `PORT` changes the default port; `RAG_ALLOWED_ORIGINS` can specify a comma-separated frontend-origin list. `RAG_AUTO_INIT=0` is for an explicitly injected notebook runtime or API testing, not a replacement for loading models.

## Verification

```bash
python -m src.prepare_deployment --check
python -m unittest discover -s tests -p test_api_contract.py -v
python -m unittest discover -s tests -p test_evidence_recovery.py -v
python -m unittest discover -s tests -p test_bert_protocol.py -v
```

The local asset check intentionally reports missing index files until they have been built on Colab. API tests use injected fixtures; they do not demonstrate GPU inference. The final inference/recovery implementation and checkpoint are the same ones used in the saved T4 evaluation. See `RELEASE_VERIFICATION.md` for the checks actually completed for this transfer.
