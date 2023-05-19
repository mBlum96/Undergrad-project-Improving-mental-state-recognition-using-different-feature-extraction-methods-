import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def one_r_feature_selector(data, target_col, threshold=0.6):
    """
    Use OneR algorithm to select features. 
    It will return all features which have accuracy equal or more than the threshold.
    """
    selected_features = []
    le = LabelEncoder()

    X = data.drop(target_col, axis=1)
    y = le.fit_transform(data[target_col])

    for feature in X.columns:
        clf = DecisionTreeClassifier(max_depth=1)  # OneR is essentially a 1-level decision tree
        scores = cross_val_score(clf, pd.DataFrame(X[feature]), y, cv=5)
        if scores.mean() >= threshold:
            selected_features.append(feature)
    
    return selected_features
