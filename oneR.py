import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data from the CSV file
data = pd.read_csv('out.csv')

# Extract the features and labels from the data
X = data.iloc[:, :-1]  # Select all but the last column
y = data.iloc[:, -1]  # Select only the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize a dictionary to store the rules and accuracies for each feature
feature_rules = {}
feature_accuracies = {}

# Iterate over each feature
for feature in X_train.columns:
    attribute_values = X_train[feature].unique()
    rules = {}

    # Iterate over each value of the attribute
    for value in attribute_values:
        class_counts = y_train[X_train[feature] == value].value_counts()
        most_frequent_class = class_counts.idxmax()
        rules[value] = most_frequent_class

    # Apply the rules to the test set and calculate accuracy
    predictions = X_test[feature].map(rules).fillna(0)
    accuracy = (predictions == y_test).mean()

    # Store the rules and accuracy for the feature
    feature_rules[feature] = rules
    feature_accuracies[feature] = accuracy

    # Create a file to store the results
    results_file = open("results.txt", "w")

    # Iterate over the features and their accuracies
    for feature, accuracy in feature_accuracies.items():
        results_file.write(f"Feature: {feature}\n")
        results_file.write(f"Accuracy: {accuracy}\n")
        results_file.write("\n")

    # Close the results file
    results_file.close()

    # Print the number of remaining features with accuracy >= 0.6
    remaining_features = [feature for feature, accuracy in feature_accuracies.items() if accuracy >= 0.4]
    results_file = open("results.txt", "a")
    results_file.write("Number of Remaining Features: " + str(len(remaining_features)) + "\n")
    results_file.close()

    
