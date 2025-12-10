# Demand Forecasting ML Pipeline

End-to-end retail demand forecasting pipeline using LightGBM, zero-inflation-aware feature engineering, and ensemble learning for robust, automated model training.

## Overview

This pipeline implements a comprehensive demand forecasting system with the following key features:

- **Time-series leak prevention**: Strict train/validation splitting that prevents data leakage across dates
- **LightGBM-based learning**: Multi-step feature selection and model training
- **Poisson/Negative Binomial regression**: Count data modeling with proper distribution assumptions
- **Ensemble learning**: Stacking of Poisson predictions as features for final LightGBM model
- **Automated pipeline**: End-to-end processing for 27 tasks (sum + h1-h26 horizons)

## Pipeline Architecture

### Step 1: Feature Importance Calculation
- Uses all features with 3-fold time-series cross-validation
- Calculates feature importance rankings using LightGBM

### Step 2: Feature Selection (K-grid CV)
- Evaluates different numbers of top features (K candidates: 60, 120, 180, 240, 300)
- Selects optimal K based on validation RMSE

### Step 3: Poisson Feature Generation
- Trains Poisson regression (with Ridge regularization) on top 60 features
- Generates Poisson predictions as additional features for final model
- Uses proper preprocessing: winsorization, standardization, and one-hot encoding

### Step 4: Final LightGBM Training
- Combines selected K features + Poisson prediction features
- Trains final LightGBM model on 80/20 time-series split

## Key Features

- **Time-series aware**: Prevents data leakage by ensuring no date overlap between train/validation
- **Zero-inflation handling**: Uses Tweedie loss and Poisson regression for count data
- **Robust preprocessing**: Winsorization, standardization, and proper handling of categorical variables
- **Comprehensive evaluation**: RMSE, MAE, sMAPE, R², and bias metrics

## Requirements

```python
numpy
pandas
lightgbm
statsmodels
scikit-learn
```

## Usage

The main pipeline is implemented in `Learn_complete_ver.ipynb`. Execute the notebook to run the full pipeline for all 27 tasks.

## Output Structure

The pipeline generates:
- Feature importance rankings
- K-grid CV results
- Poisson model predictions
- Final LightGBM models
- Evaluation metrics for all tasks

## License

See repository for license information.
