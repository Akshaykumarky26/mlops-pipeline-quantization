import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("Starting model training...")

# Load the California Housing dataset
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
y = california_housing.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model (optional, but good for verification)
r_squared = model.score(X_test, y_test)
print(f"Model trained successfully! R-squared on test set: {r_squared:.4f}")

# Save the trained model using joblib
model_path = 'linear_regression_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")