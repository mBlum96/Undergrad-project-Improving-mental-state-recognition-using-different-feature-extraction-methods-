from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def run_naive_bayes(train, test, selected_feature):
    X_train = train[[selected_feature]]
    y_train = train['Label']

    X_test = test[[selected_feature]]
    y_test = test['Label']

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    
    return classification_report(y_test, y_pred)
