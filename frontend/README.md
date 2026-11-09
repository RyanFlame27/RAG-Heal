# RAG dashboard

The active frontend is `index.html` plus `static/`. It uses the final BERT-guided API and displays initial/final responses, BERT class probabilities, signal checks, selected evidence and recovery traces.

Open the dashboard URL printed by `notebooks/05_final_pipeline_demo.ipynb`. FastAPI serves these files directly; no React build or Node installation is needed. For a separate static host, copy this folder with its `static/` subfolder, then set the backend URL in Gateway settings.

`static/benchmark-data.js` is generated from `benchmark/release_summary.json`; both describe saved evaluation results, not live runs. The UI retains the original visual style. Pinned Lucide icons and their license are bundled under `static/vendor/`; fonts have system fallbacks.

The unrelated courier React source in older copies of this folder is not part of the RAG app and is omitted from the deployment ZIP.

The console uses only `/query` for live answers. The initial box contains the
actual initial generation, and the final box contains the response selected after
BERT diagnosis and recovery. The example buttons supply only the question.
There is no injected-failure mode in the main interface. Diagnostic `/demo/*`
API endpoints remain available for explicitly controlled fault tests only.
See `../LIVE_DEMO.md` for the ordinary questions checked on the T4 session.

`static/live-examples.js` contains two selected normal-query recovery records for
the example table. Only changed, non-refusal final responses are displayed there.
The table is a curated demonstration list; aggregate benchmark statistics still
use the complete saved evaluation. An unverified final status now names the failed
or unavailable final signal instead of displaying a generic message.
