import pandas as pd
from skfeature.function.information_theoretical_based import CIFE
import numpy as np

# Load data from the CSV file
data = pd.read_csv('C:/Users/igorp/OneDrive/Desktop/Nicole/Git/eeg-feature-generation-master/out.csv')

# Separate the features and target variable
X = data.iloc[:,:-1].values
y = data.iloc[:, -1].values

def select_best_features_with_CIFE(X, y, num_features):
    # Apply CIFE to select the top features
    feature_selector = CIFE.cife(X,y)
    feature_selector.fit(X, y)
    feature_scores = feature_selector.feature_importances_

    # Rank the features by score
    feature_ranks = np.argsort(-feature_scores)

    # Select the top features
    selected_features = feature_ranks[:num_features]

    # Select the best k features using the selected feature indices
    X_best = X.iloc[:, selected_features]

    # Return the selected features
    return X_best

# Select the best 60 features using CIFE
X_best = select_best_features_with_CIFE(X, y, 60)

# Print the shape of the selected features
print("Selected features shape:", X_best.shape)
