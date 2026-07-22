# AI Audit for Health Network

ETH Zürich MAS Applied Information & Data Science — "From Data to Solutions" module, **Project 2:
AI audit for health network**.

This project goal is to work with AI models explainability and auditing: we act as an AI audit
team reviewing 7 (+1 bonus) production-candidate AI systems for a fictional hospital network
(Future-Driven Diagnostics), using LIME, SHAP, and other attribution methods to check whether each
system's high accuracy is backed by a trustworthy decision process. Full brief:
`project_FDD_audit.pdf`.

## Team

- Wojciech Bentkowski
- André Waser
- Philippe Haas

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Register the Jupyter kernel for this venv so notebooks can pick it up:

```bash
python -m ipykernel install --user --name=ai-audit-fdd --display-name="Python (ai-audit-fdd)"
```

Then open `notebooks/ai_audit_fdd.ipynb` and run cells top to bottom. All datasets and model
weights are public and auto-download on first run (no API keys needed) — the imaging fine-tuning
cells (Systems 5-7) and the first HuggingFace/TorchXRayVision downloads will take a few minutes on
CPU.

## Structure

- `project_FDD_audit.pdf` — assignment brief from the TA.
- `notebooks/ai_audit_fdd.ipynb` — the guideline notebook (originally from Colab / the course's
  [FDD-WE2-public](https://github.com/eth-fdd-fs26/FDD-WE2-public) repo). One shared notebook: fill
  in the `# 🎯 TODO` cells and each system's verdict, then the final scorecard.
- `results/` — exported figures/scorecard artifacts worth keeping, if any.

No `data/` folder: every dataset (GitHub CSV mirrors, UCI, HuggingFace, MedMNIST/Zenodo,
TorchXRayVision) auto-downloads via each library's own cache when the notebook runs — nothing is
stored in the repo.

## Running on Colab (GPU)

The notebook runs locally on CPU (see Setup above), but the imaging fine-tune cells (Systems 5-7)
and first-time model downloads are much faster on Colab's GPU runtime. Edits are made locally in
this repo; Colab is used only to execute:

1. Edit `notebooks/ai_audit_fdd.ipynb` locally as usual, then `git push` to `origin`.
2. In Colab (signed in with your ETHZ Google account): **File → Open notebook → GitHub**, paste
   `ML-ETHZ-Project/AI-audit-for-health-network`, pick the branch and
   `notebooks/ai_audit_fdd.ipynb`. Request a GPU runtime if needed (**Runtime → Change runtime
   type**).
3. After running, **File → Save a copy in GitHub** to commit the executed notebook (with outputs)
   back to the repo/branch. Pull that locally to continue editing.

This is a manual round-trip, not a live sync — push before opening in Colab, and explicitly save
back after running, so local edits and Colab runs don't silently diverge.

## Team workflow

- Don't commit directly to `main`. Work on a branch per person/system, open a PR, and get at least
  one teammate to review before merging.
- This is a single shared notebook — coordinate in advance who's filling which system's TODOs and
  verdict to avoid clashing edits on the same cells.
- Final hand-in: the completed notebook plus a short audit report with the scorecard
  (**Trustworthy** / **Inconclusive** / **Not ready** per system) and answers to the brief's
  guideline questions.
