import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv("weatherdata.csv")

# Drop rows with missing values in the location column
data.dropna(subset=["Location"], inplace=True)

# Save the location information
locations = data["Location"]

# Drop the Location column for imputation
data.drop(columns=["Location"], inplace=True)

# Impute missing values with median for all remaining columns
imputer = SimpleImputer(strategy="median")
data_imputed = imputer.fit_transform(data)
data = pd.DataFrame(data_imputed, columns=data.columns)

# Add back the Location column
data["Location"] = locations

# Reshape the 'Location' column to a 2D array
locations_array = locations.values.reshape(-1, 1)

# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

# Fit and transform the 'Location' column
locations_encoded = one_hot_encoder.fit_transform(locations_array)

# Get the categories from the OneHotEncoder
categories = one_hot_encoder.categories_[0]

# Create column names for the encoded locations
column_names = [f"Location_{category}" for category in categories]

# Convert the encoded array into a DataFrame
locations_encoded_df = pd.DataFrame(locations_encoded, columns=column_names)

# Concatenate the encoded locations with the rest of the features
locations_encoded_df = locations_encoded_df.iloc[:, 1:]
X = pd.concat([data.drop(columns=['Location', 'Precip_amount']), locations_encoded_df], axis=1)
y = data['Precip_amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), X.columns)
    ])

# Create MLPRegressor model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                         alpha=0.0001, batch_size='auto', learning_rate='constant',
                         learning_rate_init=0.001, max_iter=2000, random_state=42)

# Create pipeline with preprocessing and MLPRegressor model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', mlp_model)
])

# Train the model using Randomized Search CV
param_dist = {
    'regressor__hidden_layer_sizes': [(100, 50), (200, 100), (300, 200)],  # Hidden layer sizes
    'regressor__alpha': [0.001, 0.0001, 0.00001],  # L2 penalty (regularization term) parameter
    'regressor__learning_rate_init': [0.01, 0.001, 0.0001]  # Initial learning rate
}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=5, scoring='r2', verbose=2, n_iter=10, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Evaluate the model
y_train_pred = random_search.predict(X_train)
y_test_pred = random_search.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("Artificial Neural Network (ANN) Model with Adjusted Hyperparameters:")
print(f"Training MSE: {train_mse}")
print(f"Training R-squared: {train_r2}")
print(f"Testing MSE: {test_mse}")
print(f"Testing R-squared: {test_r2}")
print(f"Testing MAE: {test_mae}")

# Plot actual vs. predicted values for the testing set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Precipitation Amount")
plt.ylabel("Predicted Precipitation Amount")
plt.title("Actual vs. Predicted Precipitation Amount (ANN Model with Adjusted Hyperparameters)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.show()

# Plot the residual distribution
plt.figure(figsize=(8, 6))
residuals = y_test - y_test_pred
plt.hist(residuals, bins=50)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()

# Humidity vs Precipitation Amount
plt.figure(figsize=(8, 6))
plt.scatter(data['Humidity'], data['Precip_amount'])
plt.xlabel("Humidity")
plt.ylabel("Precipitation Amount")
plt.title("Humidity vs. Precipitation Amount")
plt.show()
