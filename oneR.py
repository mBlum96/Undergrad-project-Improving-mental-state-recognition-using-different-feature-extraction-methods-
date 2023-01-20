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
    pivot = pd.crosstab(data[attr], data['Label'])

    # calculate the error rate for the attribute
    error = 1 - pivot.values.trace() / data.shape[0]



    # if the error is less than the current minimum, update the minimum and best attribute
    if error < min_error:
      min_error = error
      best_attr = attr

  # return the best attribute and the minimum error
  return best_attr, min_error


def selectBestAttribute(data):
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


#main function that calls the oneR function and then use naive bayes to classify the data
#adds the attribute to the list of attributes to be used in the naive bayes classifier
#if the naive bayes classifier does not improve, don't add the attribute to the list



def main():
  # read in the data
  data = pd.read_csv('out.csv')

  # initialize list of attributes to be used in the naive bayes classifier
  attributes = []

  # initialize variable for storing minimum error
  min_error = float('inf')

  # initialize variable for storing best attribute
  best_attr = None

  # initialize variable for storing the previous error
  prev_error = float('inf')

  # initialize variable for storing the previous best attribute
  prev_best_attr = None

  # initialize variable for storing the previous best attribute
  prev_attributes = None

  # iterate until the classification does not improve
  while True:
    # call the oneR function to get the best attribute and the minimum error
    best_attr, min_error = selectBestAttribute(data)
    # print("min error is ",min_error)
    # print("curr best att is :", best_attr)

    
    # remove the best attribute from the data
    data = data.drop(best_attr, axis=1)


    # if the error is greater than the previous error, stop iterating
    if min_error > prev_error:
    #   break
        continue

    # add the best attribute to the list of attributes
    attributes.append(best_attr)

    

    # update the previous error and best attribute
    prev_error = min_error
    prev_best_attr = best_attr
    prev_attributes = attributes

    #print the attribute added and the error
    print('Attribute: {}'.format(best_attr))
    print('Error: {}'.format(min_error))


  # print the attributes and the error
  print('Attributes: {}'.format(prev_attributes))
  print('Error: {}'.format(prev_error))

  #print how many attributes were used
  print('Number of attributes used: {}'.format(len(prev_attributes)))

  # split the data into training and testing sets
  train, test = train_test_split(data, test_size=0.2)

  # separate the training and testing sets into attributes and labels
  train_attr = train.drop('Label', axis=1)
  train_label = train['Label']
  test_attr = test.drop('Label', axis=1)
  test_label = test['Label']

  # create a naive bayes classifier
  clf = GaussianNB.MultinomialNB()

  # train the classifier
  clf.fit(train_attr, train_label)

  # get the predictions
  predictions = clf.predict(test_attr)

  # calculate the accuracy
  accuracy = accuracy_score(test_label, predictions)

  # print the accuracy
  print('Accuracy: {}'.format(accuracy))

main()
