import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_features(df, features=None):
    """
    Visualize the distribution of selected features in the DataFrame using histograms and boxplots.
    
    Parameters:
    - df: DataFrame containing the data.
    - features: List of feature indices to visualize. If None, all features will be visualized.
    """
    # If no specific features are provided, visualize all features
    if features is None:
        features = list(range(df.shape[1]))

    # Plot histograms for the selected features
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Histogram for Feature {feature}')
        plt.show()

    # Plot boxplots for the selected features to check for outliers
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[feature])
        plt.title(f'Boxplot for Feature {feature}')
        plt.show()

# Example usage:
# Assuming your data is stored in a DataFrame called df
# visualize_features(df)  # This will visualize all features
# visualize_features(df, features=[0, 1, 2])  # This will visualize only the first three features
