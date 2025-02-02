"""1st doc for tests
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def workflow() -> None:
    # Get data.
    weather_data = pd.read_csv('weather_prediction_dataset.csv')

    # Basic data checks
    print("\nDataset Info:")
    print(weather_data.info())

    print("\nSummary Statistics:")
    print(weather_data.describe())

    print(f"\nNumber of duplicate rows: {weather_data.duplicated().sum()}")

    # Sample the dataset if it's too large
    if len(weather_data) > 1000:
        weather_data = weather_data.sample(frac=0.5, random_state=42)
        print(f"\nDataset reduced to {len(weather_data)} rows.")

    # See missing values.
    missing_vals(weather_data)

    # Check feature correlations.
    correlation_heatmap(weather_data)


# Visualize the missing values
def missing_vals(data: pd.DataFrame) -> None:
    plt.figure(figsize=(10,6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()


# Visualize feature correlations
def correlation_heatmap(data: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()


# Plot histograms
def histograms(data: pd.DataFrame) -> None:
    data.hist(bins=50, figsize=(20,15))
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.show()


if __name__ == "__main__":
    workflow()
