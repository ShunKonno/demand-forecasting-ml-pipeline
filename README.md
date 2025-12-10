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

## Training Data Setup

This pipeline is designed for generic store-level demand forecasting and can be applied to both weekly and daily data. The main requirements for the training data are:

### Categorical Identifiers

At minimum, the dataset should include:
- `product_code`: identifier for each product
- `store_id`: identifier for each store
- Additional categorical fields describing product attributes (e.g., tag, group_code) can also be included and are treated as categorical features

### Time Granularity

- The scheme is applicable to both weekly and daily demand data
- In this project, the data is weekly, but the same structure can be used for other frequencies

### Multi-horizon Target Construction

To predict demand for the next N time steps, the pipeline assumes:

1. You create N datasets, each with a different target:
   - For horizon i (where 1 ≤ i ≤ N), the target is the demand i steps ahead
   - In other words, each horizon has its own shifted target column

2. Additionally, you create a separate sum dataset whose target is the sum of demand over the next N steps

This gives you:
- N "point forecast" tasks (h1, h2, …, hN)
- 1 "sum forecast" task (sum of h1–hN)

### Feature Engineering

On top of the above, you can add standard time-series and categorical features, such as:
- Lag features and rolling statistics (e.g., rolling mean, trend)
- Calendar features (day of week, holidays, campaigns)
- Product/store-level attributes and encodings

This structure allows the same pipeline to be reused for multiple horizons and for both point-wise and aggregated demand forecasting.

## Why This Scheme Works Well for Store-Level Demand Forecasting

Store-level demand forecasting typically has several challenging characteristics:

### 1. Zero Inflation and Negative Bias

- When you look at sales for each (store, product) pair, many days or weeks have zero sales
- This zero inflation often causes models to develop a strong negative bias (they underestimate demand), especially when trained directly on sparse count data

### 2. Weak Correlation for Distant Horizons

- The correlation between past features and far-future targets becomes weaker as the horizon increases
- As a result, forecast accuracy naturally degrades with horizon length, and the negative bias tends to be even stronger for distant steps

### 3. Business Impact of Underestimation

- In demand forecasting for inventory and store operations, underestimation is often more damaging than overestimation, because it directly leads to stockouts and lost sales
- Therefore, controlling negative bias is especially important

This pipeline addresses these issues through several design choices:

### 1. Sum-based Scaling to Reduce Negative Bias

- In addition to predicting each horizon h₁, h₂, …, hₙ independently, the pipeline also predicts the sum of demand over the next N steps
- The sum forecast tends to be more stable and less noisy than each individual horizon, and it is less affected by zero inflation at a single time point
- By using the sum forecast to scale the individual horizon predictions, the pipeline:
  - Aligns the total predicted demand with the model's best estimate of overall future demand
  - Reduces the tendency toward negative bias across all horizons
  - This is particularly effective in store-level demand forecasting, where the main goal is to avoid systematic underestimation rather than perfectly fitting every individual zero

### 2. Horizon-specific Feature Selection with Cross-validation

- For each task (sum, h1–hN), the pipeline:
  - Computes feature importance using LightGBM with time-series cross-validation
  - Performs a K-grid search over different numbers of top features (e.g., K = 60, 120, 180, 240, 300)
  - Selects the K that achieves the best validation performance for that specific horizon
- As a result:
  - Each horizon uses a tailored subset of features, rather than a one-size-fits-all feature set
  - The model becomes less "uniform" across horizons and better adapted to the signal strength at each step
  - This leads to improved accuracy for both near-term and long-term forecasts

### 3. Ensemble with Poisson / Negative Binomial Models for Count Data

- Store-level demand is count data with zero inflation, which is not perfectly modeled by standard Gaussian assumptions
- The pipeline:
  - Trains Poisson or Negative Binomial regression models (with Ridge regularization) on a subset of informative features
  - Uses these models to generate additional prediction features (e.g., expected count from Poisson/NB)
  - These predicted counts are then fed as features into the final LightGBM model, which creates an ensemble effect:
    - The count models capture structure that is natural for zero-inflated count data
    - LightGBM combines this with rich nonlinear interactions from other features
    - In practice, this improves metrics such as R² and reduces bias

### 4. Time-series Awareness and Robust Preprocessing

- The pipeline enforces time-series-aware splits to avoid look-ahead leakage:
  - No overlapping dates between train and validation
- It also includes:
  - Winsorization to handle extreme values
  - Standardization where appropriate
  - Proper encoding for categorical variables
- This ensures that the evaluation is realistic and that the model remains stable across different time spans, products, and stores

Putting these elements together, this learning scheme has shown consistent accuracy improvements in store-level demand forecasting tasks, especially in scenarios with strong zero inflation and multi-step horizons. It is designed not just to fit historical data, but to address the specific structural challenges of retail demand forecasting: sparse sales, horizon-dependent signal strength, and the critical need to avoid systematic underestimation.

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
