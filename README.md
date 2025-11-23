## Blinkit Data Analysis & Prediction App

# About The Project

This application is a Data Analysis and Sales Prediction tool built for the Grocery/Blinkit domain. It is developed as a Capstone Project under the mentorship of the Learningt / GT Program at Rajagiri College of Social Sciences (RCSS).

The goal of this app is to provide a no-code interface for store managers to analyze sales trends and predict future item sales using Machine Learning.

# Key Features

The application guides the user through a complete End-to-End Data Science workflow:

# Data Upload:

Supports CSV file uploads.

Provides an instant preview of the raw dataset.

# Data Cleaning:

Automatically detects missing values.

Fills missing numeric data with the Median.

Fills missing categorical data with the Mode.

Removes duplicate entries.

Exploratory Data Analysis (EDA):

Correlation Heatmap: To identify relationships between features.

Distribution Plots: To check the spread of data (e.g., Item MRP).

Scatter Plots: To visualize trends (e.g., MRP vs. Sales).

# Model Training:

Users can select the Target (e.g., Sales) and Features.

Trains a Linear Regression model.

Displays performance metrics: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

# Prediction:

Interactive form to input new item details.

Generates real-time sales predictions based on the trained model.

# Tech Stack

Frontend: Streamlit (for the Web UI)

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-Learn (Linear Regression)
