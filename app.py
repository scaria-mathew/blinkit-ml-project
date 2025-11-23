import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Config ---
st.set_page_config(page_title="Blinkit Data Analysis", layout="wide")

# --- Title ---
st.title("🛍️ Blinkit Data Analysis & Prediction")
st.markdown("A simple app to analyze sales data and predict outlet sales.")

# --- Sidebar Navigation ---
# NOTE: I have commented out the local image paths below to prevent errors.
# To use images, make sure the image files are in the same folder as this script
# and use relative paths like "image.png" instead of "C:\Users\..."
# st.sidebar.image("rcss_logo.webp", width=100)
# st.sidebar.image("gt_logo.webp", width=100)

st.sidebar.image(r"images/rcss_logo.webp", width=100)
st.sidebar.image(r"images/gt_logo.png", width=100)
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["1. Upload Data", "2. Data Cleaning", "3. EDA", "4. Model Training", "5. Predict"])

# --- 1. UPLOAD DATA ---
if page == "1. Upload Data":
    st.header("1. Upload Your Dataset")
    st.info("Please upload a Blinkit/Grocery sales CSV file.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("Dataset Uploaded Successfully!")
        st.write("### Preview of Data:")
        st.dataframe(df.head())
        st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
    
    # Check if data exists in session to show it even if we switch tabs
    elif "df" in st.session_state:
        st.write("### Current Data:")
        st.dataframe(st.session_state["df"].head())

# --- 2. DATA CLEANING ---
elif page == "2. Data Cleaning":
    st.header("2. Data Cleaning")
    
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first in the 'Upload Data' tab.")
    else:
        df = st.session_state["df"]
        
        st.subheader("Missing Values (Before Cleaning)")
        st.write(df.isnull().sum())
        
        # Simple Cleaning Button
        if st.button("Clean Data (Fill Missing Values)"):
            # Fill numeric missing values with median
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Fill categorical missing values with mode (most frequent)
            categorical_cols = df.select_dtypes(include='object').columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            
            # Drop duplicates
            df = df.drop_duplicates()
            
            st.session_state["df"] = df
            st.success("✅ Data Cleaned Successfully!")
            
            st.subheader("Missing Values (After Cleaning)")
            st.write(df.isnull().sum())

# --- 3. EDA (Exploratory Data Analysis) ---
elif page == "3. EDA":
    st.header("3. Exploratory Data Analysis")
    
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        # 1. Correlation Heatmap
        st.subheader("Correlation Heatmap")
        st.write("Shows how different numerical features relate to each other.")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # 2. Distribution Plot
        st.subheader("Distribution Plot")
        col = st.selectbox("Choose a column to visualize distribution:", numeric_cols)
        fig2, ax2 = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax2, color="orange")
        ax2.set_title(f"Distribution of {col}")
        st.pyplot(fig2)
        
        # 3. Scatter Plot (Relationship)
        st.subheader("Scatter Plot (Relationship)")
        x_axis = st.selectbox("X-Axis", numeric_cols, index=0)
        y_axis = st.selectbox("Y-Axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax3, color="green")
        ax3.set_title(f"{x_axis} vs {y_axis}")
        st.pyplot(fig3)

# --- 4. MODEL TRAINING ---
elif page == "4. Model Training":
    st.header("4. Train Machine Learning Model")
    
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state["df"]
        
        # Select Target (What we want to predict)
        st.subheader("Select Features and Target")
        
        # Try to guess the target column (Sales)
        all_cols = df.columns.tolist()
        default_target_idx = 0
        if "Item_Outlet_Sales" in all_cols:
            default_target_idx = all_cols.index("Item_Outlet_Sales")
        elif "Sales" in all_cols:
            default_target_idx = all_cols.index("Sales")
            
        target = st.selectbox("Select Target Column (Output)", all_cols, index=default_target_idx)
        
        # Select Features (Inputs) - Only Numeric for Simplicity
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
            
        features = st.multiselect("Select Feature Columns (Inputs)", numeric_cols, default=numeric_cols)
        
        if st.button("Train Model"):
            if not features:
                st.error("Please select at least one feature.")
            else:
                # Prepare Data
                X = df[features]
                y = df[target]
                
                # Split Data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train Model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                st.success("✅ Model Trained Successfully!")
                st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
                
                # Save Model and Features used
                with open("model.pkl", "wb") as f:
                    pickle.dump(model, f)
                st.session_state["model_features"] = features
                st.session_state["target_name"] = target

# --- 5. PREDICT ---
elif page == "5. Predict":
    st.header("5. Predict Sales")
    
    # Check if model exists
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        features = st.session_state.get("model_features", [])
        target_name = st.session_state.get("target_name", "Value")
        st.success("Model Loaded! Enter values below:")
    except FileNotFoundError:
        st.warning("⚠️ No trained model found. Please go to 'Model Training' tab first.")
        st.stop()
        
    # Create Inputs for each feature
    input_data = {}
    st.write("### Enter details:")
    for feature in features:
        val = st.number_input(f"Enter {feature}", value=0.0)
        input_data[feature] = val
        
    if st.button("Predict Outcome"):
        # Convert input to dataframe
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        
        st.metric(label=f"Predicted {target_name}", value=f"₹ {prediction[0]:.2f}")