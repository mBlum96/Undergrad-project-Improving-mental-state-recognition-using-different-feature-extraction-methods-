from tkinter import Tk
from tkinter.filedialog import askopenfilename

# import your own modules
import testTrain
import oneR
import naiveBayes

def main():
    Tk().withdraw() 
    filename = askopenfilename()
    if filename:
        train, test = testTrain.load_and_split_data(filename)
        selected_features = oneR.one_r_feature_selector(train)
        report = naiveBayes.run_naive_bayes(train, test, selected_features)
        print(report)
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
