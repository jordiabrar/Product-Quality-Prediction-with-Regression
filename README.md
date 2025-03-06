# Product Quality Prediction using Regression in Julia

## Overview
This project demonstrates how to predict product quality using regression techniques in Julia. The model takes product-related features such as price, review count, average rating, and discount percentage to predict a quality score. 

## Technologies Used
- **Julia**: Main programming language
- **MLJ.jl**: Machine learning framework for training and evaluation
- **MLJLinearModels.jl**: Provides the linear regression model
- **DataFrames.jl**: Handles structured data
- **Plots.jl**: Used for visualization

## Installation
Ensure you have Julia installed, then add the required packages:
```julia
using Pkg
Pkg.add(["DataFrames", "Random", "Statistics", "MLJ", "MLJLinearModels", "Plots"])
```

## Running the Project
Save the script as `product_quality_prediction.jl` and run it in Julia:
```julia
include("product_quality_prediction.jl")
```

## Project Breakdown
### 1. Data Generation
- Generates synthetic product data with features:
  - `price`: Price of the product
  - `review_count`: Number of product reviews
  - `average_rating`: Average customer rating
  - `discount`: Discount percentage
- The target variable `quality` is calculated based on a simulated formula.

### 2. Model Training
- Splits data into training (80%) and testing (20%) sets.
- Uses `MLJ.jl` and `MLJLinearModels.jl` to train a linear regression model.
- Evaluates the model using RMSE (Root Mean Square Error) and R-squared metrics.

### 3. Visualization
- Creates a scatter plot comparing actual vs predicted product quality.
- Saves the visualization as `pred_vs_actual.png`.

## Output
- Model evaluation metrics (RMSE and R²) printed in the console.
- Prediction vs. actual quality plot saved as `pred_vs_actual.png`.
