import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import data_io

def main():
    data_io.resync_registry()
    data_io.ensure_local_path("data/global/sequence_training_samples.parquet")

    data = pd.read_parquet('./data/global/sequence_training_samples.parquet')

    print("data size", data.size)

if __name__ == "__main__":
    main()