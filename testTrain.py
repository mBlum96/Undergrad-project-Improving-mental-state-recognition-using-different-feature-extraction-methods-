import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filename):
    df = pd.read_csv(filename)
    train, test = train_test_split(df, test_size=0.2)
    return train, test