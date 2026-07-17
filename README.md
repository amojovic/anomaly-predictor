# Anomaly Predictor

Predicts incidents in a time-series metric using logistic regression.

The project generates a synthetic metric with injected incidents, builds
sliding-window features around each time step, and trains a logistic
regression model (with feature scaling and time-series cross-validation) to
flag whether a given moment belongs to an incident.

## Files
- `predict.py` — data generation, feature extraction, training and evaluation
- `model.json` — trained coefficients, intercept and scaler parameters
- `report.pdf` — write-up

## Run
Requires `numpy` and `scikit-learn`.

```
python predict.py
```
