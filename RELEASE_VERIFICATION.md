# Transfer verification — 24 October 2026


## Normal live page — 28 October 2026

- Removed the injected-mode controls from the main interface. Every console request uses normal `/query` generation and recovery. Preset buttons fill only the question.
- Ten ordinary questions were executed on the current Colab T4. Genuine recovery changed an initial refusal to Fr. Matthew Walsh (supported) and another to 1987 (unverified). Other outcomes and response-selection limitations are recorded in `LIVE_DEMO.md`; these checks do not replace the historical benchmark.
- Updated frontend files and documentation were applied to the active Colab and mounted Drive project without restarting the API or reloading models.
- The 45 local API, injection-endpoint, recovery and protocol checks still pass. No inference algorithm, classifier weights, report or evaluation figures were changed.

## Demo addition — 28 October 2026

- Added a labelled injected-failure mode using three stored corrupted triples. It runs the existing BERT router and recovery pipeline; no reference answer or saved final answer is supplied to inference.
- 45 local API, injection, evidence-recovery and protocol checks passed. The six new checks cover input provenance, measured/missing signals, route preservation, refusal/unverified outcomes, payload isolation, shared readiness/concurrency and unchanged normal generation.
- Desktop (1440px) and mobile (390px) browser checks passed with fixture responses, including mode changes, case selection, request bodies, loading/disabled controls, error recovery, normal benchmark buttons and trace downloads. No page errors or horizontal overflow were observed.
- No BERT training, new GPU benchmark, or report changes were made. Live recovery on the current T4 still needs to be exercised before presenting. Demo runs do not alter saved evaluation results.
- A small source-only update and same-notebook installer avoid re-uploading the model weights. The full project ZIP also contains the new mode.

## Completed locally

- 12 API integration tests passed with FastAPI 0.141.1 and Starlette 1.8.0. These cover readiness, input validation, separate refusal/unverified outcomes, nested non-finite measurements, before/after data, BERT probabilities, evidence/trace fields, GPU-request concurrency, failure recovery and serving the frontend assets.
- 15 evidence-recovery tests and 12 BERT/data/protocol tests passed: **39 local tests in total**. The API tests inject fixtures; none of these local checks claims a new end-to-end GPU evaluation.
- The accepted BERT weight file is included and matches its training manifest and the final checkpoint SHA-256.
- All existing inference, retrieval, recovery, audit and classifier modules are byte-identical to the source retained with the accepted final T4 run. Changes to existing `src/` modules are confined to the API server; `src/prepare_deployment.py` is the new deployment helper. Configuration adds a release identifier while retaining the final model/recovery defaults.
- Browser checks in headless Microsoft Edge covered 1440-pixel desktop and 390-pixel mobile widths, supported/refusal/unverified responses, API 429 errors, URL validation, keyboard dismissal, repeated submission and trace downloads. HTTP responses were injected fixtures. There were no JavaScript page errors or document-level horizontal overflow. Wide tables scroll within their containers.
- The original styling was retained. Contrast, button wrapping, mobile navigation and the decorative status pulse were adjusted after the bounded UI review. Existing font/card conventions were retained rather than redesigning the product.
- JavaScript syntax, notebook code syntax, tunnel URL matching, benchmark data consistency and Python source syntax were checked. Notebook outputs are cleared.
- The transfer is built from an explicit file list with a per-file SHA-256 manifest. The archive is read back and every file hash is compared before delivery.

## Existing GPU evidence retained

`benchmark/final_results/` contains the completed 138-case T4 regression records, 50-case validation records, adaptation summary, source-index manifest and original environment package list. Those records support the displayed measurements. The transfer update does not change the model's inference/recovery algorithm or its trained checkpoint and does not represent a fresh model experiment.

## Requires the teammate's Colab runtime

The original ZIP and local workspace contain no FAISS-index binaries or pretrained Mistral/NLI/embedding/reranker downloads. A new full GPU deployment was therefore not run on this local machine. The updated Colab notebook builds/reuses the matching index, persists it to Drive, loads the named pretrained models and starts the service. The first build needs network access and a T4; subsequent sessions can reuse the saved index. Classifier retraining and benchmark reruns are not needed for deployment.
