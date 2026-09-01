# 🍛 Dhaka Kacchi Dish Classifier

A fine-tuned Vision Transformer that identifies South Asian dishes (biryani included) from a photo — trained, evaluated, and served end-to-end as a real product, not a notebook demo.

**🔗 Live demo:** [https://classifier.dhakakacchi.com/](https://classifier.dhakakacchi.com/) — deployed on AWS behind Nginx with a free TLS certificate, live and testable by anyone
**🤗 Model:** [shadmanArko/dhaka-kacchi-dish-classifier](https://huggingface.co/shadmanArko/dhaka-kacchi-dish-classifier)

![status](https://img.shields.io/badge/status-live-brightgreen) ![model](https://img.shields.io/badge/model-ViT--base-blue) ![deployment](https://img.shields.io/badge/deployed-AWS-orange)

---

## What it does

Upload a photo of a dish, and the model predicts what it is along with a confidence score. It's built on `google/vit-base-patch16-224-in21k`, fine-tuned on the [Indian Food Images Dataset](https://huggingface.co/datasets/SKSKCulinaryAI/Indian_Food_Images_Dataset).

<p align="center">
  <img src="data/sample_grid.png" width="600" alt="Sample dish classes">
</p>

## Why this project matters

This isn't just "train a model and stop" — it's the full lifecycle a production ML system actually requires:

| Stage | What was done |
|---|---|
| **Data** | Pulled a real-world imbalanced dataset from the Hugging Face Hub, ran EDA on class distribution, image size variance, and imbalance ratio before touching a model |
| **Splitting** | Built a proper train/validation/test split with **stratified sampling** to preserve class balance — no leakage between splits |
| **Modeling strategy** | Two-stage transfer learning: first a **linear probe** (frozen backbone, train only the classifier head) to get a fast, honest baseline, then **full fine-tuning** from that verified checkpoint |
| **Evaluation** | Held-out test set (never touched during training) scored with `classification_report` + confusion matrix, plus explicit analysis of the model's worst confusions (e.g. what biryani gets mistaken for) |
| **Packaging** | Exported and pushed the final model to the **Hugging Face Hub** as a versioned, shareable artifact — not a local `.pt` file |
| **Serving** | Wrapped inference in a **FastAPI** service with CORS handling, serving both the `/predict` endpoint and the static frontend |
| **Deployment** | Shipped to **AWS**, running as a `systemd` service (auto-restarts on crash/reboot) behind **Nginx** with a **Let's Encrypt HTTPS certificate** on a custom subdomain — live and publicly testable, no "clone this repo to try it" required |
| **Product framing** | The demo page itself is a real landing page, tied to an actual business (Dhaka Kacchi), not a bare Gradio widget |

## Tech stack

- **Model:** Vision Transformer (ViT) via Hugging Face `transformers`
- **Training:** `Trainer` API, linear probe → full fine-tune, tracked with `evaluate` (accuracy)
- **Serving:** FastAPI + Uvicorn
- **Frontend:** Vanilla HTML/CSS/JS (drag-and-drop upload, live confidence bar)
- **Model hosting:** Hugging Face Hub (loaded straight into the API at startup)
- **Infra:** Deployed on AWS
- **Env/deps:** managed with `uv` (`pyproject.toml` + `uv.lock`)

## How it works, end to end

```
1. load_data.py / prepare_splits.py   → pull dataset, create stratified train/val/test splits
2. eda.py                             → class balance, image size distribution, sample grid
3. preprocessing.py                   → sanity-check the image processor pipeline
4. train_linear_probe.py              → freeze backbone, train classifier head (fast baseline)
5. train_finetune.py                  → unfreeze everything, fine-tune from the probe checkpoint
6. evaluate.py                        → score on the held-out test set, confusion matrix, error analysis
7. export_model.py / upload_to_hub.py → push the final model to Hugging Face Hub
8. api.py                             → FastAPI service loads the model from the Hub and serves predictions
9. frontend/index.html                → drag-and-drop UI calling the /predict endpoint
```

## Running it locally

```bash
# install dependencies
uv sync

# run the API (serves both the model and the frontend)
uv run uvicorn src.api:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

## Project structure

```
src/
  load_data.py          # pull dataset from the Hub
  prepare_splits.py     # stratified train/val/test split
  eda.py                 # exploratory data analysis
  preprocessing.py       # image processor sanity checks
  train_linear_probe.py  # stage 1: frozen backbone
  train_finetune.py      # stage 2: full fine-tune
  evaluate.py             # test-set evaluation + confusion matrix
  export_model.py        # export trained weights
  upload_to_hub.py       # push model to Hugging Face Hub
  api.py                  # FastAPI inference service
frontend/
  index.html              # live demo UI
```

## What I'd improve next

- Add automated tests around the API and inference pipeline
- Track experiments (learning rate, epochs) with a proper experiment tracker instead of console logs
- Add data augmentation to address class imbalance found during EDA

---

Built by **Shadman Arko** — [GitHub](https://github.com/shadmanArko)
