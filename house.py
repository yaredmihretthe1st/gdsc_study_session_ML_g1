# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 2: Create synthetic dataset
np.random.seed(42)  # For reproducibility
n_samples = 100

# Feature: Study hours (0-10 hours)
X = np.random.uniform(0, 10, n_samples).reshape(-1, 1)

# Target: Test scores (linear relationship with some noise)
# Base score of 50 + 5 points per hour of study + random noise
y = 50 + 5 * X.ravel() + np.random.normal(0, 8, n_samples)

# Create a DataFrame for better visualization
df = pd.DataFrame({'Study_Hours': X.ravel(), 'Test_Score': y})
print("First 5 rows of the dataset:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Study hours range: {df['Study_Hours'].min():.1f} to {df['Study_Hours'].max():.1f} hours")
print(f"Test scores range: {df['Test_Score'].min():.1f} to {df['Test_Score'].max():.1f} points")

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coefficients
print("\n=== Model Information ===")
print(f"Intercept (baseline score): {model.intercept_:.2f}")
print(f"Coefficient (score per study hour): {model.coef_[0]:.2f}")
print(f"Model equation: Test_Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Study_Hours")

# Step 5: Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("\n=== Model Evaluation ===")

# Training set evaluation
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("Training Set Performance:")
print(f"  Mean Squared Error (MSE): {train_mse:.2f}")
print(f"  Mean Absolute Error (MAE): {train_mae:.2f}")
print(f"  R² Score: {train_r2:.2f}")

# Testing set evaluation
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTesting Set Performance:")
print(f"  Mean Squared Error (MSE): {test_mse:.2f}")
print(f"  Mean Absolute Error (MAE): {test_mae:.2f}")
print(f"  R² Score: {test_r2:.2f}")

# Step 7: Visualization
plt.figure(figsize=(15, 5))

# Subplot 1: Training data and model
plt.subplot(1, 3, 1)
plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Study Hours')
plt.ylabel('Test Score')
plt.title('Training Data and Model Fit')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Testing data and predictions
plt.subplot(1, 3, 2)
plt.scatter(X_test, y_test, color='green', alpha=0.6, label='Testing Data')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Predictions')
plt.xlabel('Study Hours')
plt.ylabel('Test Score')
plt.title('Testing Data and Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Actual vs Predicted values
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_test_pred, color='purple', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'k--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Test Scores')
plt.ylabel('Predicted Test Scores')
plt.title('Actual vs Predicted Values (Test Set)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Make a sample prediction
sample_hours = np.array([[3.5], [7.0], [9.5]])
sample_predictions = model.predict(sample_hours)

print("\n=== Sample Predictions ===")
for hours, score in zip(sample_hours.ravel(), sample_predictions):
    print(f"Study Hours: {hours} → Predicted Test Score: {score:.1f}")

# Additional: Residual analysis (optional)
residuals = y_test - y_test_pred
print("\n=== Residual Analysis ===")
print(f"Mean of residuals: {residuals.mean():.2f}")
print(f"Standard deviation of residuals: {residuals.std():.2f}")

if abs(residuals.mean()) < 1 and residuals.std() < 10:
    print("Residual analysis suggests a good model fit!")