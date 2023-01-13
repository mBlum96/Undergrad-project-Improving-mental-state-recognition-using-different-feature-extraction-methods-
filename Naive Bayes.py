import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def oneR(data):
  # get list of attributes in the data
  attributes = list(data.columns)

  # remove label column from list of attributes
  attributes.remove('Label')

  # if no attributes are present, return an empty list and minimum error of 1
  if not attributes:
    return [], 1

  # initialize variable for storing minimum error
  min_error = float('inf')

  # initialize variable for storing best attribute
  best_attr = None

  # iterate through each attribute
  for attr in attributes:
    # skip attribute if it has already been removed from the data
    if attr not in data.columns:
      continue

    # create a pivot table of the attribute and the labels
    pivot = data[[attr, 'Label']].pivot_table(index=attr, values='Label', aggfunc='mean')

    # calculate the error rate for the attribute
    error = data[attr].map(pivot['Label']).fillna(data['Label'].mean())
    error = 1 - error.eq(data['Label']).mean()

    # if the error is less than the current minimum, update the minimum and best attribute
    if error < min_error:
      min_error = error
      best_attr = attr

  # return the best attribute and the minimum error
  return best_attr, min_error

# run naive bayes algorithm on the selected attributes, get an accuracy score, and remove the best attributes from the data
  #then run OneR again appending the best attribute to the list of selected attributes get an accuracy score, and remove the best attributes from the data
  #stop when the accuracy score stops improving
def runOneR(data):
  # initialize variable for storing selected attributes
  selected_attributes = []

  # initialize variable for storing minimum error
  min_error = float('inf')

  # initialize variable for storing best attribute
  best_attr = None

  # initialize variable for storing minimum error
  min_error = float('inf')

  # run OneR algorithm
  while True:
    # get the best attribute and the minimum error
    best_attr, min_error = oneR(data)

    # if no best attribute was found, break out of the loop
    if best_attr is None:
      break

    # append the best attribute to the list of selected attributes
    selected_attributes.append(best_attr)

    # get the accuracy score of the selected attributes
    accuracy = runNaiveBayes(data, selected_attributes)

    # if the accuracy is less than the current minimum, update the minimum and best attribute
    if accuracy < min_error:
      min_error = accuracy
      best_attr = selected_attributes

    # remove the best attribute from the data
    data = data.drop(best_attr, axis=1)

  # return the best attribute and the minimum error
  return best_attr, min_error

# run naive bayes algorithm on the selected attributes and get an accuracy score
def runNaiveBayes(data, selected_attributes):
  # get the features and labels
  features = data[selected_attributes]
  labels = data['Label']

  # split the data into training and testing sets
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)

  # initialize and train the naive bayes classifier
  clf = GaussianNB()
  clf.fit(features_train, labels_train)

  # get the prediction labels
  labels_pred = clf.predict(features_test)

  # get the accuracy score
  accuracy = accuracy_score(labels_test, labels_pred)

  # return the accuracy score
  return accuracy

# main function
def main():
  # get the data
  data = pd.read_csv('Jordan-s\eeg-feature-generation\code\out.csv')

  # run OneR algorithm
  best_attr, min_error = runOneR(data)

  # print the best attribute and the minimum error
  print('Best attribute:', best_attr)
  print('Minimum error:', min_error)




