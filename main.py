from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import csv
import time

import testTrain
import oneR
import naiveBayes

def main():
    Tk().withdraw() 
    # filename = askopenfilename()
    #file is located at /Users/michaelblum/Desktop/eeg-feature-generation/out.csv
    filename = "/Users/michaelblum/Desktop/eeg-feature-generation/out.csv"
    if filename:
        train, test = testTrain.load_and_split_data(filename)
        selected_features = oneR.one_r_feature_selector(train, 'Label')

        # Create lists to store all metrics
        total_precisions = []
        total_recalls = []
        total_f1_scores = []

        # Open output file
        with open('output.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["Feature", "Label", "Precision", "Recall", "F1-Score", "Support"])

            for feature in selected_features:
                report = naiveBayes.run_naive_bayes(train, test, feature)
                # Convert report string to dictionary
                report_dict = report_to_dict(report)


                # Write data
                for label, metrics in report_dict.items():
                    writer.writerow([feature, label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
                    # Add metrics to total lists
                    total_precisions.append(metrics['precision'])
                    total_recalls.append(metrics['recall'])
                    total_f1_scores.append(metrics['f1-score'])
                
                # Calculate and write average precision after each feature's data
                avg_precision = sum([metrics['precision'] for metrics in report_dict.values()]) / len(report_dict)
                writer.writerow([feature, "Average", avg_precision, "", "", ""])

        avg_total_precision = sum(total_precisions) / len(total_precisions)
        avg_total_recall = sum(total_recalls) / len(total_recalls)
        avg_total_f1_score = sum(total_f1_scores) / len(total_f1_scores)
        # print(f"Average precision over all features: {avg_total_precision:.4f}")
        # print(f"Average recall over all features: {avg_total_recall:.4f}")
        # print(f"Average F1-score over all features: {avg_total_f1_score:.4f}")

        # print(f"Number of selected features: {len(selected_features)}")
    return avg_total_precision
                    
def report_to_dict(report):
    """
    Convert sklearn classification report to dictionary.
    """
    report_lines = report.split('\n')
    report_dict = {}
    for line in report_lines[2:-5]:  # Skip header and summary lines
        row_data = line.split()
        label = row_data[0]
        metrics = {
            'precision': float(row_data[1]),
            'recall': float(row_data[2]),
            'f1-score': float(row_data[3]),
            'support': int(row_data[4])
        }
        report_dict[label] = metrics
    return report_dict

if __name__ == "__main__":
    main()

# New code to run main 100 times and average the results
total_elapsed_time = 0
total_accuracy = 0
num_runs = 100

for _ in range(num_runs):
  #count how many times we've run
  print(f'Run {_ + 1} of {num_runs}')

  # start timer  
  start = time.perf_counter()  

  # call the main function and get the accuracy  
  accuracy = main()  

  # stop timer and calculate elapsed time  
  elapsed = time.perf_counter() - start  

  # add to totals
  total_elapsed_time += elapsed
  total_accuracy += accuracy

# calculate averages
average_elapsed_time = total_elapsed_time / num_runs
average_accuracy = total_accuracy / num_runs

print(f'Average accuracy over {num_runs} runs: {average_accuracy}')
print(f'Average time over {num_runs} runs: {average_elapsed_time} seconds')
