# Streamlit Glass Type Predictor
The Streamlit Glass Type Predictor is a web application that allows users to explore and predict the type of glass based on its chemical composition. This application uses machine learning models to make predictions and provides various data visualization options to explore the dataset.
## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Contributing](#contributing)

## Prerequisites

Before running this application, ensure that you have the necessary Python libraries and tools installed. You can install the required packages by running the following command:

```bash
pip install numpy pandas streamlit seaborn matplotlib scikit-learn
```
## Installation
1. Clone this repository to your local machine:
```bash
https://github.com/AnkitDogra-07/glass-type.git
```
2. Change the working directory to the project folder:
```bash
cd streamlit-glass-predictor
```
3. Run the Streamlit application:
```bash
streamlit run glass-type-predictor.py
```
The Streamlit app should open in your web browser.

## Usage
Once the application is running, you can use it to perform the following tasks:
## Exploratory Data Analysis
- View the raw dataset.
- Create scatter plots, histograms, and box plots to explore the dataset's features.
## Machine Learning Model Prediction
- Select a machine learning model (SVM, Random Forest, or Logistic Regression).
- Input feature values (Refractive Index, Sodium, Magnesium, Aluminum, Silicon, Potassium, Calcium, Barium, Iron).
- Click the "Predict Glass Type" button to make a glass type prediction.
## Features
- Explore the glass dataset with various data visualization options.
- Predict the type of glass based on chemical composition.
- Choose from multiple machine learning models for prediction.
## Machine Learning Models
This application provides the following machine learning models for glass type prediction:
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression
  
You can select a model and input feature values to make predictions.
## Contributing
If you would like to contribute to this project, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, concise commit messages.
4. Push your changes to your fork.
5. Create a pull request to the main repository.
