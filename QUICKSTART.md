# Quick Start

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app runs immediately using a transparent rule-based scorecard.

## Load the trained model
```bash
python src/train_model.py
```

This writes `model/coverage_navigator_demo.joblib` and `model/metrics.json`.
Restart the app and the sidebar will confirm the trained model is loaded, and
the Model & Governance page will show real threshold and subgroup metrics.

## Regenerate the figures
```bash
python src/make_figures.py
```

Rewrites the three PNGs in `docs/images/` from the synthetic dataset.

## Publish
Full walkthrough in `GITHUB_SETUP.md`, covering the repository push, GitHub
Pages, and Streamlit Community Cloud deployment.
