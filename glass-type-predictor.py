# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache_data()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# S3.1: Create a function that accepts an ML model object say 'model' and the nine features as inputs 
# and returns the glass type.
@st.cache_data()
def pred_function(mod_nam , ri , na , mg , al , si , k , ca , ba , fe):
  glass_type = mod_nam.predict([[ri , na , mg , al , si , k , ca , ba , fe]])
  glass_type = glass_type[0]
  if glass_type == 1:
    return "building windows float processed"

  elif glass_type == 2:
    return "building windows non float processed"

  elif glass_type == 3:
    return "vehicle windows float processed"

  elif glass_type == 4:
    return "vehicle windows non float processed"

  elif glass_type == 5:
    return "containers"

  elif glass_type == 6:
    return "tableware"

  else:
    return "headlamp"
# S4.1: Add title on the main page and in the sidebar.
st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")
# S5.1: Using the 'if' statement, display raw data on the click of the checkbox.
if st.sidebar.checkbox("Show raw data"):
  st.subheader("Full Dataset")
  st.dataframe(glass_df)
# S6.1: Scatter Plot between the features and the target variable.
# Add a subheader in the sidebar with label "Scatter Plot".
st.sidebar.subheader("Scatter Plot")
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
features_list = st.sidebar.multiselect('Select the x-axis values:' , ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'), key="feature_multiselect")
# S6.2: Create scatter plots between the features and the target variable.
# Remove deprecation warning.
st.set_option('deprecation.showPyplotGlobalUse', False)
for i in features_list:
  st.subheader(f"Scatter Plot between {i} and Glass Type")
  plt.figure(figsize = (15 , 10))
  sns.scatterplot(x = glass_df[i] , y = glass_df["GlassType"])
  st.pyplot()
# S6.3: Create histograms for all the features.
# Sidebar for histograms.
st.sidebar.subheader("Histogram")
# Choosing features for histograms.
hist_features = st.sidebar.multiselect('Select the x-axis values:' , ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'), key="hist_multiselect")
# Create histograms.
for i in hist_features:
  st.subheader(f"Histogram for {i}")
  plt.figure(figsize = (15 , 10))
  plt.hist(glass_df[i] , bins = "sturges" , edgecolor = "black")
  st.pyplot()
# S6.4: Create box plots for all the columns.
# Sidebar for box plots.
st.sidebar.subheader("Boxplot")
# Choosing columns for box plots.
box_features = st.sidebar.multiselect('Select the x-axis values:' , ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'), key="box_multiselect")
# Create box plots.
for i in box_features:
  st.subheader(f"Boxplot for {i}")
  plt.figure(figsize = (15 , 10))
  sns.boxplot(glass_df[i])
  st.pyplot()  