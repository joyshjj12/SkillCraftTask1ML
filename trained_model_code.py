import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Feature engineering
train_df['TotalBath'] = (
    train_df['FullBath'] + 
    0.5 * train_df['HalfBath'] + 
    train_df['BsmtFullBath'] + 
    0.5 * train_df['BsmtHalfBath']
)

test_df['TotalBath'] = (
    test_df['FullBath'] + 
    0.5 * test_df['HalfBath'] + 
    test_df['BsmtFullBath'] + 
    0.5 * test_df['BsmtHalfBath']
)

# Additional feature engineering
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']

# Include additional features
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath', 'TotalSF', 'GarageArea', 'WoodDeckSF',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']

# Convert relevant columns to numeric types
train_df[features] = train_df[features].apply(pd.to_numeric, errors='coerce')
test_df[features] = test_df[features].apply(pd.to_numeric, errors='coerce')

# Handle missing values if any (fill NaNs with mean)
X_train = train_df[features].fillna(train_df[features].mean())
y_train = train_df['SalePrice']
X_test = test_df[features].fillna(test_df[features].mean())

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model selection and tuning (example using Gradient Boosting Regressor)
params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

gb_model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gb_model, param_grid=params, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

best_gb_model = grid_search.best_estimator_

# Evaluate the best model on the validation set
y_pred_valid = best_gb_model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred_valid)
r2 = r2_score(y_valid, y_pred_valid)

print(f"Validation Mean Squared Error: {mse}")
print(f"Validation R^2 Score: {r2}")

# Make predictions on the test set
y_pred_test = best_gb_model.predict(X_test)

# Create a DataFrame with the test predictions
test_predictions = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': y_pred_test
})

# Save the test predictions to a CSV file
test_predictions.to_csv('test_predictions.csv', index=False)
