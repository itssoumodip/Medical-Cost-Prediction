
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

try:
    data = pd.read_csv("insurance.csv")
    print("Dataset loaded successfully!")
    print("First 5 rows of the dataset:")
    print(data.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
    data = pd.DataFrame({
        'age': [19, 33, 28, 62, 46],
        'sex': ['female', 'male', 'male', 'female', 'female'],
        'bmi': [27.9, 33.77, 33.0, 26.29, 23.45],
        'children': [0, 1, 3, 0, 1],
        'smoker': ['yes', 'no', 'no', 'yes', 'no'],
        'region': ['southwest', 'southeast', 'southeast', 'northwest', 'southwest'],
        'charges': [16884.92, 1725.55, 4449.46, 27808.72, 8240.58]
    })
    print("\nLoaded a dummy dataset for demonstration purposes.")

print("\n--- Exploratory Data Analysis ---")
print("\nDataset Information:")
data.info()

print("\nDescriptive Statistics:")
print(data.describe())

plt.figure(figsize=(12, 6))
sns.histplot(data['charges'], kde=True, bins=50)
plt.title('Distribution of Medical Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Categorical Features vs. Medical Charges', fontsize=20)

sns.boxplot(ax=axes[0, 0], x='smoker', y='charges', data=data)
axes[0, 0].set_title('Smoker vs. Charges')

sns.boxplot(ax=axes[0, 1], x='sex', y='charges', data=data)
axes[0, 1].set_title('Sex vs. Charges')

sns.boxplot(ax=axes[1, 0], x='region', y='charges', data=data)
axes[1, 0].set_title('Region vs. Charges')

sns.boxplot(ax=axes[1, 1], x='children', y='charges', data=data)
axes[1, 1].set_title('Number of Children vs. Charges')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


plt.figure(figsize=(12, 6))
sns.scatterplot(x='age', y='charges', data=data, hue='smoker', palette='viridis', alpha=0.7)
plt.title('Age vs. Charges (Colored by Smoker Status)')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='bmi', y='charges', data=data, hue='smoker', palette='magma', alpha=0.7)
plt.title('BMI vs. Charges (Colored by Smoker Status)')
plt.show()

print("\n--- Feature Engineering ---")
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# Create interaction features
print("Creating interaction features...")
data['age_bmi'] = data['age'] * data['bmi']
data['age_smoker'] = data.apply(lambda x: x['age'] if x['smoker'] == 'yes' else 0, axis=1)
data['bmi_smoker'] = data.apply(lambda x: x['bmi'] if x['smoker'] == 'yes' else 0, axis=1)

# Update numerical features with interactions
numerical_features += ['age_bmi', 'age_smoker', 'bmi_smoker']

print("\n--- Data Preprocessing ---")
X = data.drop('charges', axis=1)
y = data['charges']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

print("\n--- Model Training with Cross-Validation and Hyperparameter Tuning ---")

# Define parameter grids for each model
param_grids = {
    "Linear Regression": {},
    "Ridge Regression": {'regressor__alpha': [0.1, 1.0, 10.0]},
    "Lasso Regression": {'regressor__alpha': [0.1, 1.0, 10.0]},
    "Random Forest": {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None]
    },
    "Gradient Boosting": {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1]
    }
}

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Perform k-fold cross-validation
    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='r2')
    
    results[name] = {
        'R-squared': r2,
        'MAE': mae,
        'MSE': mse,
        'CV_scores': cv_scores,
        'pipeline': best_pipeline,
        'best_params': grid_search.best_params_
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"R-squared: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")
    print(f"Cross-validation R-squared scores: {cv_scores}")
    print(f"Mean CV R-squared: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Plot residuals
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {name}')
    plt.show()
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted for {name}')
    plt.show()
    
    # For linear models, analyze coefficients
    if hasattr(best_pipeline.named_steps['regressor'], 'coef_'):
        print("\nFeature Coefficients:")
        feature_names = (numerical_features + 
                        list(best_pipeline.named_steps['preprocessor']
                             .named_transformers_['cat']
                             .get_feature_names_out(categorical_features)))
        coefficients = pd.DataFrame(
            {'feature': feature_names,
             'coefficient': best_pipeline.named_steps['regressor'].coef_}
        )
        print(coefficients.sort_values(by='coefficient', ascending=False))

print("\n--- Model Comparison ---")
results_df = pd.DataFrame({model: {'R-squared': res['R-squared'], 'MAE': res['MAE']}
                           for model, res in results.items()}).T

print(results_df.sort_values(by='R-squared', ascending=False))

best_model_name = results_df['R-squared'].idxmax()
best_model_pipeline = results[best_model_name]['pipeline']
print(f"\nBest performing model: {best_model_name}")

if hasattr(best_model_pipeline.named_steps['regressor'], 'feature_importances_'):
    print("\n--- Feature Importance ---")
    
    try:
        encoded_feature_names = best_model_pipeline.named_steps['preprocessor'] \
            .named_transformers_['cat'] \
            .get_feature_names_out(categorical_features)
        
        all_feature_names = numerical_features + list(encoded_feature_names)
        
        importances = best_model_pipeline.named_steps['regressor'].feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        print(feature_importance_df)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title(f'Feature Importance for {best_model_name}')
        plt.show()

    except Exception as e:
        print(f"Could not retrieve feature names for importance plot: {e}")


print("\n--- Making a Prediction on New Data ---")

def prepare_prediction_data(age, sex, bmi, children, smoker, region):
    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Add interaction features
    data['age_bmi'] = data['age'] * data['bmi']
    data['age_smoker'] = data.apply(lambda x: x['age'] if x['smoker'] == 'yes' else 0, axis=1)
    data['bmi_smoker'] = data.apply(lambda x: x['bmi'] if x['smoker'] == 'yes' else 0, axis=1)
    
    return data

# Test case 1: Non-smoker
new_data = prepare_prediction_data(
    age=35,
    sex='male',
    bmi=28.5,
    children=2,
    smoker='no',
    region='northeast'
)

print("New individual's data (Non-smoker):")
print(new_data)

predicted_charge = best_model_pipeline.predict(new_data)
print(f"\nPredicted Medical Charge: ${predicted_charge[0]:.2f}")

# Test case 2: Smoker
new_data_smoker = prepare_prediction_data(
    age=35,
    sex='male',
    bmi=28.5,
    children=2,
    smoker='yes',
    region='northeast'
)

print("\nNew individual's data (smoker):")
print(new_data_smoker)
predicted_charge_smoker = best_model_pipeline.predict(new_data_smoker)
print(f"\nPredicted Medical Charge (Smoker): ${predicted_charge_smoker[0]:.2f}")