# AI Audit for Health Network

3-person team project (Wojciech Bentkowski, André Waser, Philippe Haas) for the ETH Zürich MAS
Applied Information & Data Science, "From Data to Solutions" module — **Project 2: AI audit for
health network**. Not a Kaggle competition: the deliverable is a completed notebook plus a short
audit report, handed in to the Teaching Assistant. Full brief: `project_FDD_audit.pdf` in repo root.

## Scenario

Future-Driven Diagnostics (a fictional hospital network) has 7 AI systems in production-candidate
state, each reporting strong accuracy (92-97%) but never audited for *why* they work. We are hired
as the AI audit team. For every system we must answer three questions:

1. What is the model using to make its decision?
2. Does that hold up, or is it a shortcut that happens to correlate with the right answer?
3. How do we know the explanation itself is trustworthy?

## The audit loop (applied to every system, every modality)

**Look** (see the prediction) → **Explain** (find the evidence) → **Stress-test** (randomize the
model's weights and re-explain) → **Verdict** (trustworthy / inconclusive / not ready).

The stress-test is the crux of the method: an explanation of a *trained* model must fall apart once
the model's learned weights are destroyed (`randomize_weights()` in the notebook, defined once in
the Master Setup cell and reused unmodified for every system). If an attribution map barely changes
after randomization, it was never really describing the model's learned knowledge — it was
describing the input data's structure. BatchNorm/LayerNorm/GroupNorm layers are reset to identity
(not zeroed) so the randomized model still produces a live, non-constant signal to explain.

## The 7 systems (+ 1 bonus)

| # | System | Modality | Data source | Explain method(s) |
|---|--------|----------|--------------|--------------------|
| 1 | Cardiovascular risk score | Tabular | GitHub CSV mirror (cardio, 70k rows) | LIME (local) + SHAP (global) |
| 2 | Readmission predictor | Tabular | UCI Diabetes 130-US Hospitals (id=296) | LIME + SHAP (TODO: same pattern as #1) |
| 3 | Stroke triage flag | Tabular (imbalanced) | GitHub CSV mirror (~5k rows) | LIME + SHAP (TODO: same pattern as #1) |
| 4 | Chest X-ray reader | Imaging | TorchXRayVision pretrained DenseNet | 3 attribution methods side by side |
| 5 | Skin-lesion screener | Imaging | HF `marmal88/skin_cancer` (HAM10000 mirror), fine-tuned live | TODO: same pattern as #4 |
| 6 | Breast-ultrasound screener | Imaging | MedMNIST `BreastMNIST`, fine-tuned live | TODO: same pattern as #4 |
| 7 | Colon-histopathology classifier | Imaging | MedMNIST `PathMNIST`, fine-tuned live | TODO: same pattern as #4 |
| 8 (bonus) | Clinical assistant | LLM (GPT-2) | HF `gpt2` | Attention + gradient×embedding |

All data/model sources are public and auto-download (GitHub raw mirrors, `ucimlrepo`, HuggingFace
datasets/models, MedMNIST/Zenodo, TorchXRayVision) — no API keys or manual downloads needed.

Notebook source: `notebooks/ai_audit_fdd.ipynb`, originally from
[eth-fdd-fs26/FDD-WE2-public](https://github.com/eth-fdd-fs26/FDD-WE2-public) (see the notebook's
`colab` metadata for the exact provenance link).

## On startup

- Check that `.venv/` exists and `requirements.txt` is installed into it; create/install if not.
- Check that a Jupyter kernel is registered for `.venv` (`jupyter kernelspec list`); register it if
  missing (see `README.md` for the command).
- The notebook was authored for Colab with an A100 GPU. Locally it runs fine on CPU (or MPS on
  Apple Silicon) — just slower for the imaging fine-tuning cells (System 5-7) and the CXR/HF model
  downloads on first run. `DEVICE` in the Master Setup cell auto-detects CUDA vs CPU; add an MPS
  branch there if running on Apple Silicon and CPU is too slow.

## What's left to do (marked `# 🎯 TODO` in the notebook)

- System 2 (readmission): SHAP DeepExplainer + randomize-and-re-run-LIME, mirroring System 1.
- System 3 (stroke): same two TODOs, mirroring System 1.
- Systems 5-7 (skin lesion, breast ultrasound, colon histopathology): run the three attribution
  methods + the randomize/re-explain stress test, mirroring System 4's pattern.
- Every system's markdown "Your verdict" cell: fill in the actual verdict
  (**Trustworthy** / **Inconclusive** / **Not ready**) plus reasoning, once its explain + stress-test
  cells have been run.
- Final scorecard cell (near the end): aggregate all 7 (or 8) verdicts.
- Separately from the notebook: a short written audit report summarizing observations per system
  and answering the brief's guideline questions.

## Working conventions

- No `data/` folder in the repo: every dataset is fetched by the notebook's own loader functions
  (GitHub CSV mirrors, `ucimlrepo`, HuggingFace datasets/models, MedMNIST/Zenodo, TorchXRayVision)
  into each library's own cache — never commit downloaded artifacts.
- Don't commit directly to `main`; use a branch per person/experiment and a PR.
- This is one shared notebook (not one-per-person like a Kaggle repo) since the deliverable is a
  single guideline notebook with a shared verdict per system — coordinate who's filling which
  system's TODOs to avoid merge conflicts on the same cells.
- Keep `random_state=0` / `torch.manual_seed(0)` as set in the Master Setup cell so results are
  reproducible across runs and across teammates.
