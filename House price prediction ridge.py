from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Load the dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Display the first few rows of the dataset
print(df.head())

# Choose features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Ridge model with a regularization parameter (alpha)
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha value

# Fit the model
ridge_model.fit(X_train, y_train)

# Make predictions
predictions = ridge_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.scatter(y_test, predictions)
plt.xlabel("Actual Median House Values")
plt.ylabel("Predicted Median House Values")
plt.title("Actual vs Predicted House Values with Ridge Regression")
plt.show()
