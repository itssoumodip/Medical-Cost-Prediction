# Medical Cost Prediction

This repository implements a compact, reproducible pipeline to predict individual medical insurance charges using machine learning. The project includes exploratory data analysis, feature engineering, model selection (with cross-validated grid search), model comparison, diagnostic visualizations, and a demonstration of interactive prediction for new inputs. The main script is `predict_costs.py`.

Contents
- `predict_costs.py` — main script: EDA, preprocessing, training, evaluation, and example predictions
- `insurance.csv` — dataset (place in project root)
- `paper.tex` — IEEE-style LaTeX paper describing the project (ready to paste into Overleaf)
- `requirements.txt` — Python dependencies

Quick start (Windows, cmd.exe)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python predict_costs.py
```

How it works
- Loads `insurance.csv` (if missing, `predict_costs.py` will create a small demo DataFrame).
- Performs EDA: data info, descriptive statistics, and plots for distribution and relationships.
- Creates interaction features: `age_bmi`, `age_smoker`, `bmi_smoker`.
- Builds a `ColumnTransformer` + `Pipeline` to preprocess and train models.
- Runs `GridSearchCV` over multiple models (Linear, Ridge, Lasso, RandomForest, GradientBoosting) with 5-fold CV.
- Reports R-squared, MAE, MSE and visual diagnostics. Demonstrates example predictions.

Notes and next steps
- To deploy the model as a web app, add a small `app.py` using Streamlit that loads a serialized model (joblib) and provides input fields for predictions.
- For production, serialize the best model and serve over a small API (FastAPI) and add monitoring.

License
This project is provided as-is for educational purposes. Add a license file if you plan to open-source it.
# Medical-Cost-Prediction