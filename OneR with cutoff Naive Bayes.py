import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the data into a Pandas DataFrame
data = pd.read_csv('C:/Users/igorp/OneDrive/Desktop/Nicole/Git/eeg-feature-generation-master/out.csv')

# Extract the features and labels from the data
X = data.iloc[:, :-1]  # Select all but the last column
y = data.iloc[:, -1]  # Select only the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize variables to keep track of the best accuracy and the corresponding number of features
best_accuracy = 0
best_k = 0

# Try different values of k and evaluate the accuracy of the model
for k in range(1, X.shape[1] + 1):
    # Apply OneR feature selection to select the top k features
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X_train, y_train)

    # Train a Naive Bayes classifier on the selected features
    nb = GaussianNB()
    nb.fit(X_new, y_train)

    # Evaluate the accuracy of the model on the test set
    X_test_new = selector.transform(X_test)
    accuracy = nb.score(X_test_new, y_test)

    # Update the best accuracy and corresponding number of features if necessary
    if selector.scores_[selector.scores_ > 60.0].size >= best_k and accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = selector.scores_[selector.scores_ > 60.0].size

    # Print the current accuracy and number of features
    print(f"k={k}, accuracy={accuracy:.3f}")

# Print the best number of features and corresponding accuracy
print(f"Best k={best_k}, accuracy={best_accuracy:.3f}")

# Apply OneR feature selection to select the top k features
selector = SelectKBest(f_classif, k=best_k)
X_new = selector.fit_transform(X, y)

# Get the indices of the top k features
indices = selector.get_support(indices=True)

# Print the names of the top k features
feature_names = list(X.columns[indices])
print('Top features:', feature_names)