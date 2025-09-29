"""Main script for house price prediction model training and evaluation.

This script orchestrates the entire ML pipeline using modular functions from house_model.py.
To change the model, simply modify the functions in house_model.py.
"""

import pandas as pd
import logging
import argparse

from src.isc301.house_model import (
    data_preprocessing,
    model_fit,
    prepare_validation_test_data,
    evaluate_model,
    get_model_info,
    plot_model_evaluation,
    plot_model_comparison,
    plot_feature_importance,
    plot_lasso_vs_ridge_comparison
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.isc301.config import project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Load the house dataset."""
    data_path = f"{project_root}/data/maisons.csv"
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def plot_data_exploration(df: pd.DataFrame, show_plots: bool = True) -> None:
    """Create comprehensive data exploration plots."""
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))

    # Price distribution
    plt.subplot(3, 4, 1)
    plt.hist(df['prix'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Price (€)')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    plt.ticklabel_format(style='plain', axis='x')

    # Log price distribution
    plt.subplot(3, 4, 2)
    plt.hist(np.log(df['prix']), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Log Price')
    plt.ylabel('Frequency')
    plt.title('Log Price Distribution')

    # Surface vs Price scatter
    plt.subplot(3, 4, 3)
    plt.scatter(df['surf_hab'], df['prix'], alpha=0.6, color='green')
    plt.xlabel('Surface Habitable (m²)')
    plt.ylabel('Price (€)')
    plt.title('Surface vs Price')
    plt.ticklabel_format(style='plain', axis='y')

    # Surface per room vs Price
    surface_per_room = df['surf_hab'] / df['n_pieces']
    plt.subplot(3, 4, 4)
    plt.scatter(surface_per_room, df['prix'], alpha=0.6, color='orange')
    plt.xlabel('Surface per Room (m²/room)')
    plt.ylabel('Price (€)')
    plt.title('Surface per Room vs Price')
    plt.ticklabel_format(style='plain', axis='y')

    # Building type distribution
    plt.subplot(3, 4, 5)
    type_counts = df['type_batiment'].value_counts()
    plt.bar(range(len(type_counts)), type_counts.values, color='purple', alpha=0.7)
    plt.xticks(range(len(type_counts)), type_counts.index, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title('Building Type Distribution')

    # Kitchen quality distribution
    plt.subplot(3, 4, 6)
    kitchen_counts = df['qualite_cuisine'].value_counts()
    plt.bar(range(len(kitchen_counts)), kitchen_counts.values, color='brown', alpha=0.7)
    plt.xticks(range(len(kitchen_counts)), kitchen_counts.index, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title('Kitchen Quality Distribution')

    # Material quality vs Price boxplot
    plt.subplot(3, 4, 7)
    materials = sorted(df['qualite_materiau'].unique())
    price_by_material = [df[df['qualite_materiau'] == mat]['prix'].values for mat in materials]
    plt.boxplot(price_by_material, labels=materials)
    plt.xlabel('Material Quality')
    plt.ylabel('Price (€)')
    plt.title('Price by Material Quality')
    plt.xticks(rotation=45)
    plt.ticklabel_format(style='plain', axis='y')

    # Kitchen quality vs Price boxplot
    plt.subplot(3, 4, 8)
    kitchens = ['mediocre', 'moyenne', 'bonne', 'excellente']
    price_by_kitchen = [df[df['qualite_cuisine'] == kit]['prix'].values for kit in kitchens if kit in df['qualite_cuisine'].values]
    valid_kitchens = [kit for kit in kitchens if kit in df['qualite_cuisine'].values]
    plt.boxplot(price_by_kitchen, labels=valid_kitchens)
    plt.xlabel('Kitchen Quality')
    plt.ylabel('Price (€)')
    plt.title('Price by Kitchen Quality')
    plt.xticks(rotation=45)
    plt.ticklabel_format(style='plain', axis='y')

    # Number of rooms vs Price
    plt.subplot(3, 4, 9)
    plt.scatter(df['n_pieces'], df['prix'], alpha=0.6, color='red')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Price (€)')
    plt.title('Number of Rooms vs Price')
    plt.ticklabel_format(style='plain', axis='y')

    # Correlation heatmap
    plt.subplot(3, 4, 10)
    # Encode categorical variables for correlation
    df_encoded = df.copy()
    cuisine_encoder = {"mediocre": 0, "moyenne": 1, "bonne": 2, "excellente": 3}
    df_encoded["qualite_cuisine_encoded"] = df_encoded["qualite_cuisine"].map(cuisine_encoder)
    type_encoder = {"individuelle": 0, "individuelle reconvertie": 1, "duplex": 2, "bout de rangée": 3, "milieu de rangée": 4}
    df_encoded["type_batiment_encoded"] = df_encoded["type_batiment"].map(type_encoder)
    df_encoded["surface_by_room"] = df_encoded["surf_hab"] / df_encoded["n_pieces"]

    numeric_cols = ['prix', 'surf_hab', 'n_pieces', 'qualite_materiau', 'qualite_cuisine_encoded', 'type_batiment_encoded', 'surface_by_room']
    corr_matrix = df_encoded[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Price outliers visualization
    plt.subplot(3, 4, 11)
    plt.boxplot(df['prix'])
    plt.ylabel('Price (€)')
    plt.title('Price Distribution (Box Plot)')
    plt.ticklabel_format(style='plain', axis='y')

    # Surface outliers visualization
    plt.subplot(3, 4, 12)
    plt.boxplot(df['surf_hab'])
    plt.ylabel('Surface (m²)')
    plt.title('Surface Distribution (Box Plot)')

    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.savefig('data_exploration_plots.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_prediction_analysis(X_train, y_train, X_val, y_val, X_test, y_test, show_plots: bool = True) -> None:
    """Create detailed prediction analysis plots."""
    from src.isc301.house_model import model_predict

    # Make predictions
    y_train_pred = model_predict(X_train)
    y_val_pred = model_predict(X_val)
    y_test_pred = model_predict(X_test)

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))

    # Prediction accuracy by price range
    plt.subplot(3, 4, 1)
    price_ranges = [(0, 200000), (200000, 400000), (400000, 600000), (600000, float('inf'))]
    range_labels = ['<200k', '200k-400k', '400k-600k', '>600k']
    accuracies = []

    for (low, high), label in zip(price_ranges, range_labels):
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_test_pred[mask]) / y_test[mask])) * 100
            accuracies.append(100 - mape)
        else:
            accuracies.append(0)

    plt.bar(range_labels, accuracies, color='lightblue', alpha=0.7)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Price Range (€)')
    plt.title('Prediction Accuracy by Price Range')
    plt.xticks(rotation=45)

    # Residuals vs Features
    feature_names = ["surface_by_room", "qualite_materiau", "surf_hab"]
    residuals_test = y_test - y_test_pred

    for i, feature_name in enumerate(feature_names):
        plt.subplot(3, 4, i + 2)
        plt.scatter(X_test[:, i], residuals_test, alpha=0.6, color='red')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel(feature_name.replace('_', ' ').title())
        plt.ylabel('Residuals (€)')
        plt.title(f'Residuals vs {feature_name.replace("_", " ").title()}')
        plt.ticklabel_format(style='plain', axis='y')

    # Prediction error distribution
    plt.subplot(3, 4, 5)
    errors = y_test - y_test_pred
    plt.hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Prediction Error (€)')
    plt.ylabel('Frequency')
    plt.title('Test Set: Prediction Error Distribution')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.ticklabel_format(style='plain', axis='x')

    # Relative error distribution
    plt.subplot(3, 4, 6)
    relative_errors = (y_test - y_test_pred) / y_test * 100
    plt.hist(relative_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('Test Set: Relative Error Distribution')
    plt.axvline(x=0, color='red', linestyle='--')

    # Prediction intervals
    plt.subplot(3, 4, 7)
    sorted_indices = np.argsort(y_test)
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_test_pred[sorted_indices]

    plt.plot(y_test_sorted, y_test_sorted, 'r--', label='Perfect Prediction', linewidth=2)
    plt.scatter(y_test_sorted, y_pred_sorted, alpha=0.6, color='blue', s=20)

    # Add confidence bands (±20% error)
    plt.fill_between(y_test_sorted, y_test_sorted * 0.8, y_test_sorted * 1.2,
                     alpha=0.2, color='gray', label='±20% Error Band')

    plt.xlabel('Actual Price (€)')
    plt.ylabel('Predicted Price (€)')
    plt.title('Test Set: Predictions with Error Bands')
    plt.legend()
    plt.ticklabel_format(style='plain')

    # Model performance across datasets
    plt.subplot(3, 4, 8)
    from sklearn.metrics import r2_score, mean_absolute_error

    datasets = ['Training', 'Validation', 'Test']
    r2_scores = [
        r2_score(y_train, y_train_pred),
        r2_score(y_val, y_val_pred),
        r2_score(y_test, y_test_pred)
    ]
    mae_scores = [
        mean_absolute_error(y_train, y_train_pred),
        mean_absolute_error(y_val, y_val_pred),
        mean_absolute_error(y_test, y_test_pred)
    ]

    x_pos = np.arange(len(datasets))

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x_pos - 0.2, r2_scores, 0.4, label='R² Score', color='skyblue', alpha=0.7)
    bars2 = ax2.bar(x_pos + 0.2, mae_scores, 0.4, label='MAE (€)', color='lightcoral', alpha=0.7)

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('R² Score', color='blue')
    ax2.set_ylabel('MAE (€)', color='red')
    ax1.set_title('Model Performance Across Datasets')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets)

    # Add value labels
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

    for bar, score in zip(bars2, mae_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{score:,.0f}', ha='center', va='bottom')

    # Learning curve simulation
    plt.subplot(3, 4, 9)
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_r2_mean = []
    val_r2_mean = []

    for size in train_sizes:
        n_samples = int(size * len(X_train))
        if n_samples > 10:  # Minimum samples for training
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            X_subset = X_train[indices]
            y_subset = y_train[indices]

            # Quick model training
            from sklearn.linear_model import LinearRegression
            temp_model = LinearRegression()
            temp_model.fit(X_subset, np.log(y_subset))

            train_pred = np.exp(temp_model.predict(X_subset))
            val_pred = np.exp(temp_model.predict(X_val))

            train_r2_mean.append(r2_score(y_subset, train_pred))
            val_r2_mean.append(r2_score(y_val, val_pred))
        else:
            train_r2_mean.append(0)
            val_r2_mean.append(0)

    plt.plot(train_sizes, train_r2_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, val_r2_mean, 'o-', color='red', label='Validation Score')
    plt.xlabel('Training Set Size (Fraction)')
    plt.ylabel('R² Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Price prediction ranges
    plt.subplot(3, 4, 10)
    price_bins = np.linspace(y_test.min(), y_test.max(), 20)
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    bin_errors = []

    for i in range(len(price_bins) - 1):
        mask = (y_test >= price_bins[i]) & (y_test < price_bins[i + 1])
        if mask.sum() > 0:
            bin_errors.append(np.mean(np.abs(y_test[mask] - y_test_pred[mask])))
        else:
            bin_errors.append(0)

    plt.plot(bin_centers, bin_errors, 'o-', color='purple', alpha=0.7)
    plt.xlabel('Actual Price (€)')
    plt.ylabel('Mean Absolute Error (€)')
    plt.title('Prediction Error vs Price Range')
    plt.ticklabel_format(style='plain')
    plt.grid(True, alpha=0.3)

    # Outlier analysis
    plt.subplot(3, 4, 11)
    errors_abs = np.abs(y_test - y_test_pred)
    outlier_threshold = np.percentile(errors_abs, 95)  # Top 5% errors
    outliers = errors_abs > outlier_threshold

    plt.scatter(y_test[~outliers], y_test_pred[~outliers], alpha=0.6, color='blue', s=20, label='Normal')
    plt.scatter(y_test[outliers], y_test_pred[outliers], alpha=0.8, color='red', s=50, label='Outliers (Top 5%)')

    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel('Actual Price (€)')
    plt.ylabel('Predicted Price (€)')
    plt.title('Prediction Outliers Analysis')
    plt.legend()
    plt.ticklabel_format(style='plain')

    # Feature contribution to errors
    plt.subplot(3, 4, 12)
    feature_error_corr = []
    for i, feature_name in enumerate(feature_names):
        corr = np.corrcoef(X_test[:, i], np.abs(residuals_test))[0, 1]
        feature_error_corr.append(abs(corr) if not np.isnan(corr) else 0)

    plt.bar(range(len(feature_names)), feature_error_corr, color='orange', alpha=0.7)
    plt.xticks(range(len(feature_names)), [name.replace('_', ' ').title() for name in feature_names], rotation=45)
    plt.ylabel('Correlation with Abs Error')
    plt.title('Feature Correlation with Prediction Errors')

    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.savefig('prediction_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()


def main(show_plots: bool = False, model_type: str = "auto"):
    """Main training and evaluation pipeline."""
    logger.info("Starting house price prediction pipeline...")

    # Step 1: Load data
    df = load_data()

    # Step 2: Data preprocessing
    logger.info("Starting data preprocessing...")
    X_train, y_train = data_preprocessing(df)
    logger.info(
        f"Preprocessing complete. Training set shape: X={X_train.shape}, y={y_train.shape}"
    )

    # Step 3: Model training
    logger.info("Training model...")
    model = model_fit(X_train, y_train, model_type=model_type)
    logger.info("Model training completed")

    # Step 4: Get model information
    model_info = get_model_info()
    logger.info(f"Selected model: {model_info['selected_model']}")
    logger.info(f"Model type: {model_info['model_type']}")
    logger.info(f"Cross-validation R²: {model_info['cv_score']:.4f}")

    # Display coefficients if available
    if 'coefficients' in model_info:
        logger.info("Model coefficients (original features):")
        for feat, coef in zip(model_info['feature_names'],
                             model_info['coefficients'][:len(model_info['feature_names'])]):
            logger.info(f"  {feat}: {coef:.4f}")
        if 'intercept' in model_info:
            logger.info(f"  intercept: {model_info['intercept']:.4f}")

    # Step 5: Prepare validation and test data
    logger.info("Preparing validation and test datasets...")
    X_val, y_val, X_test, y_test = prepare_validation_test_data(df)
    logger.info(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

    # Step 6: Evaluate on all datasets
    train_metrics = evaluate_model(X_train, y_train, "Training")
    val_metrics = evaluate_model(X_val, y_val, "Validation")
    test_metrics = evaluate_model(X_test, y_test, "Test")

    # Step 7: Generate plots if requested
    if show_plots:
        logger.info("Generating comprehensive visualization suite...")

        # Data exploration plots
        logger.info("Creating data exploration plots...")
        plot_data_exploration(df, show_plots=show_plots)

        # Model comparison plot
        logger.info("Creating model comparison plot...")
        plot_model_comparison(show_plots=show_plots)

        # Model evaluation plots (actual vs predicted, residuals, etc.)
        logger.info("Creating model evaluation plots...")
        plot_model_evaluation(X_train, y_train, X_val, y_val, X_test, y_test,
                             show_plots=show_plots)

        # Feature importance plot
        logger.info("Creating feature importance plot...")
        plot_feature_importance(X_train, show_plots=show_plots)

        # Advanced prediction analysis
        logger.info("Creating prediction analysis plots...")
        plot_prediction_analysis(X_train, y_train, X_val, y_val, X_test, y_test,
                                show_plots=show_plots)

        # Lasso vs Ridge detailed comparison
        logger.info("Creating Lasso vs Ridge comparison plots...")
        plot_lasso_vs_ridge_comparison(X_train, y_train, show_plots=show_plots)

        logger.info("All plots generated successfully!")
        logger.info("Generated plot files:")
        logger.info("  - data_exploration_plots.png")
        logger.info("  - model_comparison_plots.png")
        logger.info("  - model_evaluation_plots.png")
        logger.info("  - feature_importance_plots.png")
        logger.info("  - prediction_analysis_plots.png")
        logger.info("  - lasso_vs_ridge_comparison.png")
    else:
        logger.info("Plots disabled. Use --show-plots to enable comprehensive visualization.")

    # Step 8: Summary
    logger.info("=" * 50)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Selected Model: {model_info['selected_model']} ({model_info['model_type']})")
    logger.info(f"Cross-validation R²: {model_info['cv_score']:.4f}")
    logger.info(f"Training R²:   {train_metrics['r2']:.4f}")
    logger.info(f"Validation R²: {val_metrics['r2']:.4f}")
    logger.info(f"Test R²:       {test_metrics['r2']:.4f}")
    logger.info("=" * 50)
    logger.info("Pipeline completed successfully!")

    return {
        "model_info": model_info,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "data": {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="House price prediction with polynomial models and regularization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with auto model selection, no plots
  python main.py --show-plots              # Run with plots enabled
  python main.py --model lasso             # Use Lasso model specifically
  python main.py --model ridge --show-plots # Use Ridge model with plots
        """
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display evaluation plots (actual vs predicted, residuals, feature importance)"
    )
    parser.add_argument(
        "--model",
        choices=["auto", "linear", "lasso", "ridge"],
        default="auto",
        help="Model type to use (default: auto - selects best performing model)"
    )

    args = parser.parse_args()

    logger.info(f"Running with model={args.model}, show_plots={args.show_plots}")
    results = main(show_plots=args.show_plots, model_type=args.model)