# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import plot_roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
data_path = 'dementia_patients_health_data.csv'
data = pd.read_csv(data_path)
data = data.drop(['Prescription', 'Dosage in mg', 'Cognitive_Test_Scores'], axis=1) # Remove directly related variables
data.columns = [col.lower() for col in data.columns]

# Function to preprocess data
def preprocess_data(df):
    
    X = df.drop('dementia', axis=1)
    y = df['dementia']

    # Handle categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, y

def create_models():
    models = {
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'LASSO': LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        'Neural Network': MLPClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    return models

import warnings
warnings.filterwarnings('ignore')

# Function to train and evaluate models
def train_evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability for ROC AUC
        
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        # Print results
        # print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
        
    return results

# Split the data
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create models and evaluate them
models = create_models()
results = train_evaluate_models(models, X_train, X_test, y_train, y_test)
results_df = pd.DataFrame(results).T
print(results_df)

# Define the hyperparameter grids for each model
param_grids = {
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
    'Random Forest': {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
    'LASSO': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']},
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100), (50, 100, 50)], 'alpha': [0.0001, 0.001, 0.01, 0.1], 'learning_rate_init': [0.0001, 0.001, 0.01], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd'], 'max_iter': [200, 500]},
    'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 9]}
}

# Function to tune, evaluate models, and return best models for plotting
def tune_and_cross_validate_models(models, param_grids, X, y):
    results = []
    best_models = {}  # Dictionary to store the best-tuned models
    
    for name, model in models.items():
        print(f"Tuning and cross-validating {name}...")

        # Set up GridSearchCV for cross-validation and hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        
        # Get the best model after cross-validation tuning
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Calculate evaluation metrics based on cross-validated predictions
        y_pred = cross_val_predict(best_model, X, y, cv=5, method="predict")
        y_prob = cross_val_predict(best_model, X, y, cv=5, method="predict_proba")[:, 1]
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)

        # Store the results in a list
        results.append({
            'Model': name,
            'Best Parameters': grid_search.best_params_,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        })

        # Print the metrics and the best parameters for each model
        # print(f"{name} - Best Parameters: {grid_search.best_params_}")
        # print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
    
    # Convert the results list into a DataFrame
    results_df = pd.DataFrame(results)
    return best_models, results_df

# Create, tune and evaluate models, returning the best models and their evaluation results
models = create_models()
best_models, cross_val_results_df = tune_and_cross_validate_models(models, param_grids, X, y)

# Function to plot ROC curves in a 3x2 grid for older sklearn versions
def plot_roc_curves(models, X_test, y_test):
    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models.items()):
        plot_roc_curve(model, X_test, y_test, ax=axes[idx], name=name)
        axes[idx].plot([0, 1], [0, 1], 'k--', lw=2)
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].set_title(f'ROC Curve: {name}')
    
    # Hide any unused subplots if there are fewer models
    for i in range(len(models), 6):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Plot ROC curves for all models
plot_roc_curves(best_models, X_test, y_test)

# Function to plot confusion matrices in a 3x2 grid
def plot_confusion_matrices(models, X_test, y_test):
    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models.items()):
        # Predict and create confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot the confusion matrix on the grid
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=axes[idx], cmap=plt.cm.Blues, values_format='g')
        axes[idx].set_title(f'Confusion Matrix: {name}')
    
    # Hide any unused subplots if there are fewer models
    for i in range(len(models), 6):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Plot confusion matrices for all models
plot_confusion_matrices(best_models, X_test, y_test)

# Function to plot feature importances in a 3x2 grid, skipping models without feature importances
def plot_feature_importances(models, feature_names):
    # Count the number of models with feature importances or coefficients
    valid_models = [(name, model) for name, model in models.items() if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')]
    
    # Calculate the grid size based on the number of valid models
    n_models = len(valid_models)
    n_rows = (n_models + 1) // 2

    # Create a grid of subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
    axes = axes.ravel() if n_models > 1 else [axes]

    for idx, (name, model) in enumerate(valid_models):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Plot the feature importances on the grid
            axes[idx].bar(range(len(importances)), importances[indices], align='center')
            axes[idx].set_xticks(range(len(importances)))
            axes[idx].set_xticklabels([feature_names[i] for i in indices], rotation=90)
            axes[idx].set_title(f'Feature Importances: {name}')
            axes[idx].set_ylabel('Importance')

        elif hasattr(model, 'coef_'):
            importances = model.coef_[0]
            indices = np.argsort(importances)[::-1]

            # Plot the feature importances for models with coef_
            axes[idx].bar(range(len(importances)), importances[indices], align='center')
            axes[idx].set_xticks(range(len(importances)))
            axes[idx].set_xticklabels([feature_names[i] for i in indices], rotation=90)
            axes[idx].set_title(f'Feature Importances: {name}')
            axes[idx].set_ylabel('Importance')

    # Remove any unused subplots
    for i in range(len(valid_models), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Plot feature importances
feature_names = pd.get_dummies(data, drop_first=True).drop('dementia', axis=1).columns
plot_feature_importances(best_models, feature_names)

# Function to calculate relative feature contributions and plot a pie chart
def plot_average_feature_importances(models, feature_names, threshold=0.03):
    total_importances = np.zeros(len(feature_names))
    valid_model_count = 0
    
    # Accumulate feature importances from models
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):  # For models with feature_importances_
            importances = model.feature_importances_
            total_importances += importances / np.sum(importances)  # Normalize and accumulate
            valid_model_count += 1

        elif hasattr(model, 'coef_'):  # For models with coef_
            importances = np.abs(model.coef_[0])
            total_importances += importances / np.sum(importances)  # Normalize and accumulate
            valid_model_count += 1
    
    if valid_model_count > 0:
        average_importances = total_importances / valid_model_count

        # Combine contributions for specific categories
        combined_features = {
            'Smoking Status': [i for i, name in enumerate(feature_names) if 'smoking_status_' in name],
            'Depression Status': [i for i, name in enumerate(feature_names) if 'depression_status_' in name],
            'Education Level': [i for i, name in enumerate(feature_names) if 'education_level' in name],
            'APOE ε4': [i for i, name in enumerate(feature_names) if 'apoe_ε4' in name]
        }
        
        combined_importances = {}
        for category, indices in combined_features.items():
            combined_importances[category] = np.sum(average_importances[indices])

        # Create the final importance list with combined categories included
        remaining_indices = [i for i in range(len(feature_names)) if not any(i in indices for indices in combined_features.values())]
        final_importances = list(average_importances[remaining_indices])
        final_labels = list(feature_names[remaining_indices])

        for category, importance in combined_importances.items():
            if importance > 0:
                final_importances.append(importance)
                final_labels.append(category)

        # Apply the thresholding (sum features below 3% into 'Others')
        others_importance = np.sum([imp for imp in final_importances if imp < threshold])
        significant_indices = [i for i, imp in enumerate(final_importances) if imp >= threshold]

        final_importances = [final_importances[i] for i in significant_indices]
        final_labels = [final_labels[i] for i in significant_indices]

        if others_importance > 0:
            final_importances.append(others_importance)
            final_labels.append('Others')

        # Create a pie chart with the average feature importances
        plt.figure(figsize=(10, 8))
        plt.pie(final_importances, labels=final_labels, autopct='%1.1f%%', startangle=0)
        plt.axis('equal')
        plt.show()
    else:
        print("No models with feature importances or coefficients were found.")

feature_names = pd.get_dummies(data, drop_first=True).drop('dementia', axis=1).columns
plot_average_feature_importances(best_models, feature_names)
