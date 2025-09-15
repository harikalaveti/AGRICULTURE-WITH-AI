# pesticide_model.py
# Trains RandomForest model for pesticide dose recommendation
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def load_pesticide_data():
    """Load pesticide recommendation data from CSV file"""
    csv_path = "backend/dataset/pesticide.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} pesticide records from {csv_path}")
    else:
        print(f"Warning: {csv_path} not found. Generating synthetic data...")
        # Generate synthetic pesticide data
        np.random.seed(RANDOM_STATE)
        
        crops = ['wheat', 'rice', 'maize', 'tomato', 'potato']
        disease_severities = [1, 2, 3, 4, 5]
        
        data = []
        for _ in range(1000):
            crop = np.random.choice(crops)
            disease_severity = np.random.choice(disease_severities)
            humidity = np.random.uniform(40, 90)
            rainfall = np.random.exponential(5)
            
            # Generate realistic dose based on features
            base_dose = 1.0
            severity_factor = disease_severity * 0.5
            humidity_factor = (humidity - 50) / 100 * 0.5
            rainfall_factor = min(rainfall / 20, 1.0) * 0.5
            
            recommended_dose = base_dose + severity_factor + humidity_factor + rainfall_factor
            recommended_dose += np.random.normal(0, 0.1)  # Add noise
            recommended_dose = max(0.5, recommended_dose)  # Ensure positive
            
            data.append({
                'crop': crop,
                'disease_severity': disease_severity,
                'humidity': humidity,
                'rainfall': rainfall,
                'recommended_dose': recommended_dose
            })
        
        df = pd.DataFrame(data)
    
    return df

def prepare_features(df):
    """Prepare features for training"""
    # Encode categorical variables
    le_crop = LabelEncoder()
    df['crop_encoded'] = le_crop.fit_transform(df['crop'])
    
    # Feature columns
    feature_cols = ['crop_encoded', 'disease_severity', 'humidity', 'rainfall']
    X = df[feature_cols]
    y = df['recommended_dose']
    
    return X, y, le_crop

def train_pesticide_model():
    """Train the pesticide dose recommendation RandomForest model"""
    print("Starting pesticide model training...")
    
    # Load data
    df = load_pesticide_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset info:")
    print(df.describe())
    
    # Prepare features
    X, y, le_crop = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Hyperparameter tuning
    print("Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    best_model = grid_search.best_estimator_
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Evaluate model
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"Training - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"Test - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Feature importance
    feature_names = ['crop_encoded', 'disease_severity', 'humidity', 'rainfall']
    feature_importance = best_model.feature_importances_
    
    print(f"\nFeature Importance:")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.4f}")
    
    # Save model and label encoder
    model_path = "models/pesticide_model.pkl"
    encoder_path = "models/crop_encoder.pkl"
    
    joblib.dump(best_model, model_path)
    joblib.dump(le_crop, encoder_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Label encoder saved to: {encoder_path}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Feature importance plot
    plt.subplot(1, 3, 1)
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    
    # Training predictions vs actual
    plt.subplot(1, 3, 2)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Dose')
    plt.ylabel('Predicted Dose')
    plt.title('Training Set: Predicted vs Actual')
    
    # Test predictions vs actual
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Dose')
    plt.ylabel('Predicted Dose')
    plt.title('Test Set: Predicted vs Actual')
    
    plt.tight_layout()
    plt.savefig('models/pesticide_model_analysis.png')
    plt.show()
    
    # Residuals plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Dose')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig('models/pesticide_residuals.png')
    plt.show()
    
    return best_model, le_crop

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)
    
    # Train model
    model, le_crop = train_pesticide_model()
    print("Pesticide model training completed successfully!")
