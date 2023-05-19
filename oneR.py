import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def one_r_feature_selector(data, target_col, max_error=0.43):
    """
    Use OneR algorithm to select features.
    Only features with an error rate less than the maximum allowable error are added.
    """
    selected_features = []
    le = LabelEncoder()

    X = data.drop(target_col, axis=1)
    y = le.fit_transform(data[target_col])

    for feature in X.columns:
        clf = DecisionTreeClassifier(max_depth=1)  # OneR is essentially a 1-level decision tree
        scores = cross_val_score(clf, pd.DataFrame(X[feature]), y, cv=5, scoring='accuracy')
        error_rate = 1.0 - scores.mean()

        if error_rate < max_error:
            selected_features.append(feature)
            print(f"Selected feature: {feature}, Error rate: {error_rate:.4f}")

    return selected_features
