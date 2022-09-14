import yaml
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier 
  
 def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        cols = df_train.columns
        cols = cols[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        # Check for data sanity
        assert (len(train_features.iloc[0]) == 132)
        assert (len(train_labels) == train_features.shape[0])

        return train_features, train_labels, df_train

    # Function to Load Test Dataset
def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        cols = df_test.columns
        cols = cols[:-1]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        # Check for data sanity
        assert (len(test_features.iloc[0]) == 132)
        assert (len(test_labels) == test_features.shape[0])

        return test_features, test_labels, df_test
