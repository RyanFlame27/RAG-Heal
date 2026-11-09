# Notebook entry points

Use **05_final_pipeline_demo.ipynb** for final deployment. It keeps the original Drive/Colab/T4 workflow and starts the updated HTML frontend and REST API through one tunnel URL.

**01_build_index.ipynb** performs only the index preparation/cache steps if you want to run them separately.

The older 02/03/04 research notebooks remain in the original workspace, but are excluded from this deployment package because their setup/training paths precede the final runtime. Evaluation and training source modules remain available under `benchmark/` and `src/classifier/`. The saved final results are in `benchmark/final_results/`; deployment does not need to run training or evaluation again.
