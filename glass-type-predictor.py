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
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    for i in df.columns:
      columns_dict[i] = column_headers[i - 1]
    df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Spliting data into features (X) and target (y)
X = glass_df.iloc[:, :-1]
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Creating a function for glass type prediction
@st.cache_data()
def pred_function(mod_nam , ri , na , mg , al , si , k , ca , ba , fe):
  # Using the provided feature values to make predictions
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
  
  
# Adding a title to the main page and in the sidebar.
st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")


# Displaying raw data on the click of the checkbox
if st.sidebar.checkbox("Show raw data"):
  st.subheader("Full Dataset")
  st.dataframe(glass_df)
  
  
# Scatter Plot between the features and the target variable
st.sidebar.subheader("Scatter Plot")
features_list = st.sidebar.multiselect('Select the x-axis values:' , ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'), key="feature_multiselect")
st.set_option('deprecation.showPyplotGlobalUse', False)
for i in features_list:
  st.subheader(f"Scatter Plot between {i} and Glass Type")
  plt.figure(figsize = (15 , 10))
  sns.scatterplot(x = glass_df[i] , y = glass_df["GlassType"])
  st.pyplot()
  
  
# Histograms for the features
st.sidebar.subheader("Histogram")
hist_features = st.sidebar.multiselect('Select the x-axis values:' , ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'), key="hist_multiselect")
for i in hist_features:
  st.subheader(f"Histogram for {i}")
  plt.figure(figsize = (15 , 10))
  plt.hist(glass_df[i] , bins = "sturges" , edgecolor = "black")
  st.pyplot()


# Box Plots for the columns
st.sidebar.subheader("Boxplot")
box_features = st.sidebar.multiselect('Select the x-axis values:' , ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'), key="box_multiselect")
for i in box_features:
  st.subheader(f"Boxplot for {i}")
  plt.figure(figsize = (15 , 10))
  sns.boxplot(glass_df[i])
  st.pyplot()  
  
# Machine Learning Model Selection
st.sidebar.subheader("Machine Learning Model")
selected_model = st.sidebar.selectbox("Select a Machine Learning Model", ("Support Vector Machine (SVM)", "Random Forest", "Logistic Regression"))
# Input Feature Values
st.sidebar.subheader("Input Feature Values")
ri = st.sidebar.number_input("Refractive Index (RI)", value=0.0, step=0.01)
na = st.sidebar.number_input("Sodium (Na)", value=0.0, step=0.01)
mg = st.sidebar.number_input("Magnesium (Mg)", value=0.0, step=0.01)
al = st.sidebar.number_input("Aluminum (Al)", value=0.0, step=0.01)
si = st.sidebar.number_input("Silicon (Si)", value=0.0, step=0.01)
k = st.sidebar.number_input("Potassium (K)", value=0.0, step=0.01)
ca = st.sidebar.number_input("Calcium (Ca)", value=0.0, step=0.01)
ba = st.sidebar.number_input("Barium (Ba)", value=0.0, step=0.01)
fe = st.sidebar.number_input("Iron (Fe)", value=0.0, step=0.01)

# Creating an empty placeholder to display the glass type prediction
glass_type_prediction = ""

# A dictionary to map model names to their corresponding classifiers
models = {
    "Support Vector Machine (SVM)": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Getting the selected classifier based on the user's choice.