import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

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


def main():
  # read in data from out.csv file
  data = pd.read_csv('out.csv')

  # initialize list to store selected attributes
  selected_attributes = []

  # initialize variable for storing previous accuracy of naive bayes model
  prev_accuracy = 0

  # continue looping until the classification accuracy of naive bayes stops improving
  while True:
    # call oneR algorithm to find best attribute
    best_attr, min_error = oneR(data)

    # add best attribute to list of selected attributes
    selected_attributes.append(best_attr)

    # create a subset of the data with only the selected attributes and the label column
    subset = data[selected_attributes + ['Label']]

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(subset.drop('Label', axis=1), subset['Label'], test_size=0.2)

    # create and fit a naive bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # predict labels for the test set
    y_pred = model.predict(X_test)

    # calculate the classification accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # if the accuracy has not improved, break out of the loop
    if accuracy <= prev_accuracy:
      break

    # update the previous accuracy
    prev_accuracy = accuracy

    # remove the selected attribute from the data
    data = data.drop(best_attr, axis=1)

  # print the selected attributes and final accuracy
  print(f'Selected attributes: {selected_attributes}')
  print(f'Final accuracy: {accuracy}')

# start timer
start = time.perf_counter()

# call the main function
main()

# stop timer and print elapsed time
elapsed = time.perf_counter() - start
print(f'Elapsed time: {elapsed:0.4f} seconds')
