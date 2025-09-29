"""House price prediction model module.

This module contains functions for data preprocessing, model fitting, and prediction
for the house price analysis project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Union, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables to store the trained model and preprocessing info
_trained_model = None
_model_info = None
_scaler = None
_poly_features = None
logger = logging.getLogger(__name__)


def data_preprocessing(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the house data for machine learning.

    Args:
        df: Raw dataframe with house data

    Returns:
        Tuple of (X, y) where X is features array and y is target array
    """
    # Create a copy to avoid modifying original data
    df_processed = df.copy()

    # Encode categorical variables
    cuisine_encoder = {"mediocre": 0, "moyenne": 1, "bonne": 2, "excellente": 3}
    df_processed["qualite_cuisine_encoded"] = df_processed["qualite_cuisine"].map(
        cuisine_encoder
    )

    type_encoder = {
        "individuelle": 0,
        "individuelle reconvertie": 1,
        "duplex": 2,
        "bout de rangée": 3,
        "milieu de rangée": 4,
    }
    df_processed["type_batiment_encoded"] = df_processed["type_batiment"].map(
        type_encoder
    )

    # Feature engineering: surface per room
    df_processed["surface_by_room"] = (
        df_processed["surf_hab"] / df_processed["n_pieces"]
    )

    # Split data
    train_df, test_df = train_test_split(df_processed, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

    # Remove outliers from training and validation sets
    upper_bound = 600000
    train_df = train_df[train_df["prix"] <= upper_bound]
    val_df = val_df[val_df["prix"] <= upper_bound]

    # Select features based on notebook analysis
    selected_features = ["surface_by_room", "qualite_materiau", "surf_hab"]

    # Prepare training data
    X_train = train_df[selected_features].values
    y_train = train_df["prix"].values

    return X_train, y_train


def model_fit(X: np.ndarray, y: np.ndarray, model_type: str = "auto") -> Union[Pipeline, LinearRegression]:
    """
    Fit polynomial models with regularization to the training data.
    Uses log transformation to ensure positive predictions.

    Args:
        X: Feature array
        y: Target array
        model_type: Type of model to fit ("linear", "lasso", "ridge", "auto")

    Returns:
        Trained model or pipeline
    """
    global _trained_model, _model_info, _scaler, _poly_features

    # Transform target to log space to ensure positive predictions
    y_log = np.log(y)

    models = {}

    # Linear regression baseline
    linear_model = LinearRegression()
    linear_model.fit(X, y_log)
    linear_score = cross_val_score(linear_model, X, y_log, cv=5, scoring='r2').mean()
    models['linear'] = {'model': linear_model, 'score': linear_score, 'type': 'LinearRegression'}

    # Polynomial features with Lasso (with proper hyperparameter tuning)
    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('lasso', Lasso(max_iter=10000))
    ])

    # Grid search for optimal Lasso alpha
    lasso_param_grid = {'lasso__alpha': [0.1, 1, 10, 100, 500]}
    lasso_grid = GridSearchCV(lasso_pipeline, lasso_param_grid, cv=3, scoring='r2', n_jobs=-1)
    lasso_grid.fit(X, y_log)

    # Get best Lasso model and score
    best_lasso = lasso_grid.best_estimator_
    lasso_score = cross_val_score(best_lasso, X, y_log, cv=5, scoring='r2').mean()
    models['lasso'] = {
        'model': best_lasso,
        'score': lasso_score,
        'type': 'PolynomialLasso',
        'best_alpha': lasso_grid.best_params_['lasso__alpha']
    }

    # Polynomial features with Ridge (with proper hyperparameter tuning)
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge())
    ])

    # Grid search for optimal Ridge alpha
    ridge_param_grid = {'ridge__alpha': [0.1, 1, 10, 100, 500, 1000]}
    ridge_grid = GridSearchCV(ridge_pipeline, ridge_param_grid, cv=3, scoring='r2', n_jobs=-1)
    ridge_grid.fit(X, y_log)

    # Get best Ridge model and score
    best_ridge = ridge_grid.best_estimator_
    ridge_score = cross_val_score(best_ridge, X, y_log, cv=5, scoring='r2').mean()
    models['ridge'] = {
        'model': best_ridge,
        'score': ridge_score,
        'type': 'PolynomialRidge',
        'best_alpha': ridge_grid.best_params_['ridge__alpha']
    }

    # Select best model or use specified type
    if model_type == "auto":
        best_model_name = max(models.keys(), key=lambda k: models[k]['score'])
        best_model_info = models[best_model_name]
        logger.info(f"Auto-selected best model: {best_model_name} with CV R² = {best_model_info['score']:.4f}")
    else:
        if model_type not in models:
            raise ValueError(f"Invalid model_type. Choose from: {list(models.keys())} or 'auto'")
        best_model_info = models[model_type]
        best_model_name = model_type
        logger.info(f"Using specified model: {model_type} with CV R² = {best_model_info['score']:.4f}")

    # Store the best model and info globally
    _trained_model = best_model_info['model']
    _model_info = {
        'selected_model': best_model_name,
        'model_type': best_model_info['type'],
        'cv_score': best_model_info['score'],
        'all_scores': {k: v['score'] for k, v in models.items()},
        'hyperparameters': {k: v.get('best_alpha', 'N/A') for k, v in models.items()}
    }

    # Log all model scores for comparison
    logger.info("Model comparison (CV R² scores and optimal hyperparameters):")
    for name, info in models.items():
        alpha_info = f" (α={info.get('best_alpha', 'N/A')})" if 'best_alpha' in info else ""
        logger.info(f"  {name}: {info['score']:.4f}{alpha_info}")

    return _trained_model


def model_predict(X: np.ndarray) -> np.ndarray:
    """
    Make predictions using the trained model.
    Automatically transforms predictions from log space back to original scale.

    Args:
        X: Feature array for prediction

    Returns:
        Predicted values array (guaranteed to be positive)
    """
    global _trained_model

    if _trained_model is None:
        raise ValueError("Model has not been fitted yet. Call model_fit() first.")

    # Predict in log space
    log_predictions = _trained_model.predict(X)

    # Transform back to original scale (exponential of log predictions)
    predictions = np.exp(log_predictions)

    return predictions


def prepare_validation_test_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare validation and test datasets with the same preprocessing as training.

    Args:
        df: Raw dataframe with house data

    Returns:
        Tuple of (X_val, y_val, X_test, y_test)
    """
    # Apply same preprocessing steps as in data_preprocessing
    df_processed = df.copy()

    # Encode categorical variables
    cuisine_encoder = {"mediocre": 0, "moyenne": 1, "bonne": 2, "excellente": 3}
    df_processed["qualite_cuisine_encoded"] = df_processed["qualite_cuisine"].map(
        cuisine_encoder
    )

    type_encoder = {
        "individuelle": 0,
        "individuelle reconvertie": 1,
        "duplex": 2,
        "bout de rangée": 3,
        "milieu de rangée": 4,
    }
    df_processed["type_batiment_encoded"] = df_processed["type_batiment"].map(
        type_encoder
    )

    # Feature engineering
    df_processed["surface_by_room"] = (
        df_processed["surf_hab"] / df_processed["n_pieces"]
    )

    # Split data (same as preprocessing)
    train_df, test_df = train_test_split(df_processed, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

    # Remove outliers from validation set
    upper_bound = 600000
    val_df = val_df[val_df["prix"] <= upper_bound]

    # Note: Test set keeps outliers as in the notebook

    # Select features
    selected_features = ["surface_by_room", "qualite_materiau", "surf_hab"]

    X_val = val_df[selected_features].values
    y_val = val_df["prix"].values

    X_test = test_df[selected_features].values
    y_test = test_df["prix"].values

    return X_val, y_val, X_test, y_test


def evaluate_model(X: np.ndarray, y_true: np.ndarray, set_name: str) -> Dict[str, float]:
    """
    Evaluate model performance on a given dataset.

    Args:
        X: Feature array
        y_true: True target values
        set_name: Name of the dataset (for logging)

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating model on {set_name} set...")

    # Make predictions
    y_pred = model_predict(X)

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    # Log metrics
    logger.info(f"{set_name} Set Results:")
    logger.info(f"  MSE (price): {mse:,.2f}")
    logger.info(f"  RMSE (price): {rmse:,.2f}")
    logger.info(f"  MAE (price): {mae:,.2f}")
    logger.info(f"  R² (price): {r2:.4f}")

    return metrics


def get_model_info() -> Dict[str, any]:
    """
    Get information about the trained model.

    Returns:
        Dictionary with model information including coefficients and feature names
    """
    global _trained_model, _model_info

    if _trained_model is None or _model_info is None:
        raise ValueError("Model has not been fitted yet. Call model_fit() first.")

    feature_names = ["surface_by_room", "qualite_materiau", "surf_hab"]

    info = {
        "selected_model": _model_info['selected_model'],
        "model_type": _model_info['model_type'],
        "cv_score": _model_info['cv_score'],
        "all_scores": _model_info['all_scores'],
        "feature_names": feature_names,
        "n_features": len(feature_names)
    }

    # Add model-specific coefficients if available
    if hasattr(_trained_model, 'coef_'):
        info["coefficients"] = _trained_model.coef_
        info["intercept"] = _trained_model.intercept_
    elif hasattr(_trained_model, 'named_steps'):
        # For pipeline models, get the final estimator's coefficients
        final_estimator = _trained_model.named_steps[list(_trained_model.named_steps.keys())[-1]]
        if hasattr(final_estimator, 'coef_'):
            info["coefficients"] = final_estimator.coef_
            info["intercept"] = final_estimator.intercept_
            # Also get polynomial feature names if available
            if 'poly' in _trained_model.named_steps:
                poly_features = _trained_model.named_steps['poly']
                info["polynomial_feature_names"] = poly_features.get_feature_names_out(feature_names)

    return info


def plot_model_evaluation(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         show_plots: bool = True) -> None:
    """
    Create comprehensive plots for model evaluation.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        show_plots: Whether to display plots
    """
    if _trained_model is None:
        raise ValueError("Model has not been fitted yet. Call model_fit() first.")

    # Make predictions
    y_train_pred = model_predict(X_train)
    y_val_pred = model_predict(X_val)
    y_test_pred = model_predict(X_test)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))

    # 1. Actual vs Predicted scatter plots
    ax1 = plt.subplot(3, 3, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training')
    min_val, max_val = min(y_train.min(), y_train_pred.min()), max(y_train.max(), y_train_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Training Set: Actual vs Predicted')
    plt.legend()

    ax2 = plt.subplot(3, 3, 2)
    plt.scatter(y_val, y_val_pred, alpha=0.6, color='green', label='Validation')
    min_val, max_val = min(y_val.min(), y_val_pred.min()), max(y_val.max(), y_val_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Validation Set: Actual vs Predicted')
    plt.legend()

    ax3 = plt.subplot(3, 3, 3)
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='orange', label='Test')
    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Test Set: Actual vs Predicted')
    plt.legend()

    # 2. Residual plots
    ax4 = plt.subplot(3, 3, 4)
    residuals_train = y_train - y_train_pred
    plt.scatter(y_train_pred, residuals_train, alpha=0.6, color='blue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Training Set: Residuals vs Predicted')

    ax5 = plt.subplot(3, 3, 5)
    residuals_val = y_val - y_val_pred
    plt.scatter(y_val_pred, residuals_val, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Validation Set: Residuals vs Predicted')

    ax6 = plt.subplot(3, 3, 6)
    residuals_test = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals_test, alpha=0.6, color='orange')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Test Set: Residuals vs Predicted')

    # 3. Residual distributions
    ax7 = plt.subplot(3, 3, 7)
    plt.hist(residuals_train, bins=30, alpha=0.7, color='blue', density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Training Set: Residual Distribution')

    ax8 = plt.subplot(3, 3, 8)
    plt.hist(residuals_val, bins=30, alpha=0.7, color='green', density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Validation Set: Residual Distribution')

    ax9 = plt.subplot(3, 3, 9)
    plt.hist(residuals_test, bins=30, alpha=0.7, color='orange', density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Test Set: Residual Distribution')

    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_model_comparison(show_plots: bool = True) -> None:
    """
    Plot comparison of different model performances.

    Args:
        show_plots: Whether to display plots
    """
    if _model_info is None:
        raise ValueError("Model has not been fitted yet. Call model_fit() first.")

    scores = _model_info['all_scores']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot of CV scores
    models = list(scores.keys())
    cv_scores = list(scores.values())
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    bars = ax1.bar(models, cv_scores, color=colors[:len(models)])
    ax1.set_ylabel('Cross-Validation R² Score')
    ax1.set_title('Model Comparison: CV R² Scores')
    ax1.set_ylim(0, max(cv_scores) * 1.1)

    # Add value labels on bars
    for bar, score in zip(bars, cv_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}', ha='center', va='bottom')

    # Highlight selected model
    selected_idx = models.index(_model_info['selected_model'])
    bars[selected_idx].set_color('gold')
    bars[selected_idx].set_edgecolor('black')
    bars[selected_idx].set_linewidth(2)

    # Model complexity visualization
    complexity_scores = {
        'linear': 1,
        'lasso': 2,
        'ridge': 2
    }

    ax2.scatter([complexity_scores[model] for model in models], cv_scores,
               s=200, c=colors[:len(models)], alpha=0.7)
    ax2.set_xlabel('Model Complexity')
    ax2.set_ylabel('Cross-Validation R² Score')
    ax2.set_title('Model Performance vs Complexity')
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Linear', 'Polynomial + Regularization'])

    # Annotate points
    for i, model in enumerate(models):
        ax2.annotate(model, (complexity_scores[model], cv_scores[i]),
                    xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_feature_importance(X: np.ndarray, show_plots: bool = True) -> None:
    """
    Plot feature importance or coefficients.

    Args:
        X: Feature array for scaling reference
        show_plots: Whether to display plots
    """
    if _trained_model is None or _model_info is None:
        raise ValueError("Model has not been fitted yet. Call model_fit() first.")

    model_info = get_model_info()

    plt.figure(figsize=(12, 8))

    if 'coefficients' in model_info:
        coeffs = model_info['coefficients']
        feature_names = model_info['feature_names']

        # For polynomial models, show top features
        if 'polynomial_feature_names' in model_info:
            poly_names = model_info['polynomial_feature_names']
            # Get top 15 features by absolute coefficient value
            abs_coeffs = np.abs(coeffs)
            top_indices = np.argsort(abs_coeffs)[-15:]

            top_coeffs = coeffs[top_indices]
            top_names = [poly_names[i] for i in top_indices]

            plt.subplot(2, 1, 1)
            colors = ['red' if c < 0 else 'blue' for c in top_coeffs]
            bars = plt.barh(range(len(top_coeffs)), top_coeffs, color=colors, alpha=0.7)
            plt.yticks(range(len(top_coeffs)), top_names)
            plt.xlabel('Coefficient Value')
            plt.title(f'Top 15 Polynomial Features ({model_info["model_type"]})')
            plt.grid(axis='x', alpha=0.3)

            # Original features only
            plt.subplot(2, 1, 2)
            orig_coeffs = coeffs[:len(feature_names)]
        else:
            orig_coeffs = coeffs

        colors = ['red' if c < 0 else 'blue' for c in orig_coeffs]
        bars = plt.barh(range(len(orig_coeffs)), orig_coeffs, color=colors, alpha=0.7)
        plt.yticks(range(len(orig_coeffs)), feature_names)
        plt.xlabel('Coefficient Value')
        plt.title(f'Original Features ({model_info["model_type"]})')
        plt.grid(axis='x', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Coefficients not available for this model type',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance')

    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.savefig('feature_importance_plots.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_lasso_vs_ridge_comparison(X: np.ndarray, y: np.ndarray, show_plots: bool = True) -> None:
    """
    Create detailed comparison plots between Lasso and Ridge models.

    Args:
        X: Feature array
        y: Target array
        show_plots: Whether to display plots
    """
    # Transform target to log space
    y_log = np.log(y)

    # Create base pipeline
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])

    # Fit base pipeline to get scaled polynomial features
    X_poly = base_pipeline.fit_transform(X)

    # Test different alpha values
    alphas = np.logspace(-3, 3, 50)  # From 0.001 to 1000

    lasso_scores = []
    ridge_scores = []
    lasso_n_features = []
    ridge_n_features = []

    for alpha in alphas:
        # Lasso
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso_cv_scores = cross_val_score(lasso, X_poly, y_log, cv=5, scoring='r2')
        lasso_scores.append(lasso_cv_scores.mean())

        # Fit to count non-zero features
        lasso.fit(X_poly, y_log)
        lasso_n_features.append(np.sum(lasso.coef_ != 0))

        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge_cv_scores = cross_val_score(ridge, X_poly, y_log, cv=5, scoring='r2')
        ridge_scores.append(ridge_cv_scores.mean())

        # Ridge doesn't zero out features, so count all
        ridge.fit(X_poly, y_log)
        ridge_n_features.append(len(ridge.coef_))

    # Create comprehensive comparison plots
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))

    # 1. CV Score vs Alpha
    plt.subplot(3, 3, 1)
    plt.semilogx(alphas, lasso_scores, 'b-', label='Lasso', linewidth=2)
    plt.semilogx(alphas, ridge_scores, 'r-', label='Ridge', linewidth=2)
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Cross-Validation R² Score')
    plt.title('Model Performance vs Regularization Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Number of features vs Alpha
    plt.subplot(3, 3, 2)
    plt.semilogx(alphas, lasso_n_features, 'b-', label='Lasso', linewidth=2)
    plt.semilogx(alphas, ridge_n_features, 'r-', label='Ridge', linewidth=2)
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Number of Non-zero Features')
    plt.title('Feature Selection vs Regularization')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Regularization path for Lasso
    plt.subplot(3, 3, 3)
    lasso_path = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_poly, y_log)
        lasso_path.append(lasso.coef_)

    lasso_path = np.array(lasso_path).T
    for i in range(min(10, lasso_path.shape[0])):  # Show top 10 features
        plt.semilogx(alphas, lasso_path[i], linewidth=1)

    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso Regularization Path (Top 10 Features)')
    plt.grid(True, alpha=0.3)

    # 4. Regularization path for Ridge
    plt.subplot(3, 3, 4)
    ridge_path = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_poly, y_log)
        ridge_path.append(ridge.coef_)

    ridge_path = np.array(ridge_path).T
    for i in range(min(10, ridge_path.shape[0])):  # Show top 10 features
        plt.semilogx(alphas, ridge_path[i], linewidth=1)

    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Coefficient Value')
    plt.title('Ridge Regularization Path (Top 10 Features)')
    plt.grid(True, alpha=0.3)

    # 5. Optimal models comparison
    if _model_info and 'hyperparameters' in _model_info:
        plt.subplot(3, 3, 5)

        # Get optimal alphas from our model training
        lasso_alpha = _model_info['hyperparameters'].get('lasso', 1.0)
        ridge_alpha = _model_info['hyperparameters'].get('ridge', 1.0)

        # Train optimal models
        lasso_opt = Lasso(alpha=lasso_alpha, max_iter=10000)
        ridge_opt = Ridge(alpha=ridge_alpha)

        lasso_opt.fit(X_poly, y_log)
        ridge_opt.fit(X_poly, y_log)

        # Compare coefficients
        feature_names = [f'Feature_{i}' for i in range(min(15, len(lasso_opt.coef_)))]
        x_pos = np.arange(len(feature_names))

        width = 0.35
        plt.bar(x_pos - width/2, lasso_opt.coef_[:len(feature_names)], width,
                label=f'Lasso (α={lasso_alpha})', alpha=0.7, color='blue')
        plt.bar(x_pos + width/2, ridge_opt.coef_[:len(feature_names)], width,
                label=f'Ridge (α={ridge_alpha})', alpha=0.7, color='red')

        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.title('Optimal Models: Coefficient Comparison')
        plt.xticks(x_pos, feature_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 6. Bias-Variance trade-off visualization
    plt.subplot(3, 3, 6)
    # Calculate approximate bias-variance using bootstrap
    n_bootstrap = 20
    lasso_preds = []
    ridge_preds = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X_poly[indices]
        y_boot = y_log[indices]

        # Train models
        lasso_boot = Lasso(alpha=1.0, max_iter=10000)  # Use reasonable alpha
        ridge_boot = Ridge(alpha=1.0)

        lasso_boot.fit(X_boot, y_boot)
        ridge_boot.fit(X_boot, y_boot)

        # Predictions on original data
        lasso_preds.append(lasso_boot.predict(X_poly))
        ridge_preds.append(ridge_boot.predict(X_poly))

    # Calculate variance
    lasso_var = np.var(lasso_preds, axis=0).mean()
    ridge_var = np.var(ridge_preds, axis=0).mean()

    plt.bar(['Lasso', 'Ridge'], [lasso_var, ridge_var],
            color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Prediction Variance')
    plt.title('Model Variance Comparison')
    plt.grid(True, alpha=0.3)

    # 7. Feature importance comparison
    plt.subplot(3, 3, 7)
    if _model_info and 'hyperparameters' in _model_info:
        lasso_alpha = _model_info['hyperparameters'].get('lasso', 1.0)
        ridge_alpha = _model_info['hyperparameters'].get('ridge', 1.0)

        lasso_opt = Lasso(alpha=lasso_alpha, max_iter=10000)
        ridge_opt = Ridge(alpha=ridge_alpha)

        lasso_opt.fit(X_poly, y_log)
        ridge_opt.fit(X_poly, y_log)

        # Feature importance (absolute coefficients)
        lasso_importance = np.abs(lasso_opt.coef_)
        ridge_importance = np.abs(ridge_opt.coef_)

        # Top features
        top_indices = np.argsort(ridge_importance)[-10:]

        x_pos = np.arange(len(top_indices))
        width = 0.35

        plt.barh(x_pos - width/2, lasso_importance[top_indices], width,
                label='Lasso', alpha=0.7, color='blue')
        plt.barh(x_pos + width/2, ridge_importance[top_indices], width,
                label='Ridge', alpha=0.7, color='red')

        plt.ylabel('Top 10 Features')
        plt.xlabel('Absolute Coefficient Value')
        plt.title('Feature Importance Comparison')
        plt.yticks(x_pos, [f'F_{i}' for i in top_indices])
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 8. Model selection criteria
    plt.subplot(3, 3, 8)

    # Find best performance for each model
    best_lasso_score = max(lasso_scores)
    best_ridge_score = max(ridge_scores)
    best_lasso_alpha = alphas[np.argmax(lasso_scores)]
    best_ridge_alpha = alphas[np.argmax(ridge_scores)]

    models = ['Lasso', 'Ridge']
    scores = [best_lasso_score, best_ridge_score]
    colors = ['blue', 'red']

    bars = plt.bar(models, scores, color=colors, alpha=0.7)
    plt.ylabel('Best CV R² Score')
    plt.title('Best Performance Comparison')

    # Add value labels
    for bar, score, alpha_val in zip(bars, scores, [best_lasso_alpha, best_ridge_alpha]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}\\n(α={alpha_val:.3f})',
                ha='center', va='bottom', fontsize=10)

    plt.grid(True, alpha=0.3)

    # 9. Sparsity comparison
    plt.subplot(3, 3, 9)

    # Calculate sparsity (percentage of zero coefficients) for different alphas
    lasso_sparsity = [(len(coef) - np.count_nonzero(coef)) / len(coef) * 100
                      for coef in lasso_path.T]

    plt.semilogx(alphas, lasso_sparsity, 'b-', linewidth=2, label='Lasso')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Ridge (No Sparsity)')

    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Sparsity (%)')
    plt.title('Model Sparsity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.savefig('lasso_vs_ridge_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

