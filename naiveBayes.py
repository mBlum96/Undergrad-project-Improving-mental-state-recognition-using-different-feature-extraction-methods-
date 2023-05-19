from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def run_naive_bayes(train, test, selected_features):
    X_train = train[selected_features]
    y_train = train['target']

    X_test = test[selected_features]
    y_test = test['target']

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    return classification_report(y_test, y_pred)