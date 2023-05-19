from ruleset import OneR

def one_r_feature_selector(df, min_accuracy=0.6):
    selected_features = []
    for col in df.columns:
        if col != 'target':
            X = df[[col]]
            y = df['target']
            clf = OneR()
            clf.fit(X, y)
            accuracy = clf.score(X, y)
            if accuracy >= min_accuracy:
                selected_features.append(col)
    return selected_features