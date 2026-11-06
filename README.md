# 📈 Advertising Sales Prediction – Linear Regression

Predicts **sales revenue** from advertising spend on **TV, Radio and Newspaper** (200+ records) and turns the
model into marketing budget-allocation advice.

**Stack:** Python · Pandas · NumPy · Scikit-learn · Matplotlib

## Pipeline
1. **Data cleaning**: drop duplicates, median-impute missing spend values.
2. **Outlier detection**: IQR rule (1.5 × IQR) on every column.
3. **EDA & correlation**: scatter plots with trend lines + correlation heatmap.
4. **Model**: `LinearRegression`, 80/20 split, 5-fold cross-validation.
5. **Evaluation**: R², RMSE, MAE; residual and actual-vs-predicted diagnostics.
6. **Feature analysis**: coefficients, standardized betas, permutation importance.
7. **Business insight**: translates coefficients into budget guidance.

## Run
```bash
pip install -r requirements.txt
python -m src.generate_data     # creates data/advertising.csv (skip if using the real dataset)
python -m src.analysis          # runs everything; saves plots + report.json to outputs/
python -m pytest -q
```
**Using the real Kaggle "Advertising" dataset?** Drop it in as `data/advertising.csv`
(columns `TV, Radio, Newspaper, Sales`) and re-run `python -m src.analysis`.

## Results (included dataset is synthetic, modelled on the classic Advertising data)
| Metric | Value |
|---|---|
| Test R² | ~0.885 |
| Test RMSE | ~1.66 |
| 5-fold CV R² | ~0.884 |

Correlation with Sales: TV 0.79 · Radio 0.48 · Newspaper 0.19.
Fitted: `Sales ≈ 3.19 + 0.047·TV + 0.176·Radio − 0.001·Newspaper`.

## Key insights
- **TV is the strongest predictor.** It has the highest standardized coefficient and permutation importance, so it explains the most variation in sales.
- **Radio has the highest return per extra dollar.** Each additional $1k adds about 0.18k units vs about 0.05k for TV, so marginal budget is best shifted towards Radio.
- **Newspaper contributes ~0.** Its coefficient is negligible once TV and Radio are in the model, so cut or cap it.

## Limitations
Linear model: it ignores diminishing returns and TV×Radio synergy. Adding an interaction term or
trying ridge/gradient boosting would be sensible next steps. Correlation ≠ causation: spend levels
were not randomized.
