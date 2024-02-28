# importing all the necessary libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# creating the custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns, copy = True, with_mean = True, with_std = True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X, y = None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self
    
    def transform(self, X, y = None, copy = None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis = 1)[init_col_order]

# creating the special class that will be used to predict new data
class AbsenteeismModel():

    def __init__(self, model_file, scaler_file):

        # reading the model and the scaler files
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None

    # loading the *.csv file and preprocessing it
    def load_and_clean_data(self, data_file):

        # importing the data
        df = pd.read_csv(data_file, delimiter = ',')
 
        # dropping the ID column
        df = df.drop(['ID'], axis = 1)

        # splitting the Reason for Absence column into Multiple Dummy Variables
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)

        # grouping the Reason columns into 4 groups
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis = 1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis = 1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis = 1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis = 1)

        # dropping the Reason for Absence column
        df = df.drop(['Reason for Absence'], axis = 1)

        # attaching the 4 newly created Series to the df DataFrame
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)

        # assigning names to the newly added columns
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education',
                        'Children', 'Pet', 'Absenteeism Time in Hours', 'Reason_1',
                        'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names

        # reordering the columns in df
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date',
                                  'Transportation Expense', 'Distance to Work', 'Age',
                                  'Daily Work Load Average', 'Body Mass Index', 'Education',
                                  'Children', 'Pet', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]

        # converting the values of the Date column from string to timestamp
        df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')

        # extracting the month value from the Date column
        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)
        
        # creating the Month Value column and inserting it into df
        df['Month Value'] = list_months

        # dropping the Date column
        df = df.drop(['Date'], axis = 1)

        # reordering the columns in df
        column_names_updated = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value',
                                'Transportation Expense', 'Distance to Work', 'Age',
                                'Daily Work Load Average', 'Body Mass Index', 'Education',
                                'Children', 'Pet', 'Absenteeism Time in Hours']
        df = df[column_names_updated]

        # transforming the Education column into a Dummy Variable
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

        # dropping the variables that we decide we do not need
        df = df.drop(['Distance to Work', 'Daily Work Load Average', 'Absenteeism Time in Hours'], axis=1)

        # final checkpoints necessary before proceeding with the predictions
        self.preprocessed_data = df.copy()
        self.data = self.scaler.transform(df)

    # creating a method that outputs the probability
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred
        
    # creating a method that outputs the prediction
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
    
    # adding the Probability and the Prediction columns to the preprocessed data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

# creating an instance of the AbsenteeismModel class
absenteeism_model = AbsenteeismModel('model','scaler')
# calling the methods from the AbsenteeismModel class
absenteeism_model.load_and_clean_data('Absenteeism_data.csv')
print(absenteeism_model.predicted_outputs())