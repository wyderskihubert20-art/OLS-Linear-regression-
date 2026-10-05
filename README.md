# OLS-Linear-regression-
Custom Ordinary Least Squares (OLS) Linear Regression Pipeline

## Project Overview

The primary objective of this project is to deconstruct and implement the mathematical foundation of simple linear regression. Rather than importing pre-built models, this pipeline manually computes slope, intercept, residual errors, evaluation metrics, and professional data visualizations using only foundational data science tools (`pandas`, `numpy`, and `matplotlib`).

### Key Highlights:
* **Algorithmic Transparency:** Slope and intercept formulas are coded using covariance and variance principles.
* **Complete Evaluation Suite:** Custom-built calculations for **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and the **Coefficient of Determination ($R^2$)**.
* **Modular Architecture:** Clean separation of concerns featuring dedicated functions for data ingestion, training, evaluation, inference, and visualization.
* **Dual-Phase Execution:** Evaluates training performance on historical data before executing inference on unseen future test sets.

The model solves for the best-fit line $y = mx + b$ 
by minimizing the sum of squared residuals:
  Optimal Slope ($m$): Calculated using the formula for sample covariance divided by sample variance:
    $$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$Optimal Intercept ($b$): 
  Derived using the means of $x$ and $y$:$$b = \bar{y} - m\bar{x}$$Coefficient of Determination ($R^2$): 
    Measures the proportion of variance in the dependent variable explained by the independent variable:$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$Model Performance & ResultsEvaluated on the training dataset, the custom OLS model demonstrates exceptional precision:Training RMSE: 1.1180Training $R^2$ Score: 0.9900 (Explains 99% of variance)
