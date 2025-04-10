import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
import datetime


st.title("Neural Network Model Builder for Prediction")

st.sidebar.header("Home")


st.sidebar.markdown("[Application](http://localhost:3000/)")


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    customerloan = pd.read_csv(uploaded_file)
    
    st.write("Dataset Preview:")
    st.write(customerloan.head())
    
    columns = customerloan.columns.tolist()
    target_column = st.selectbox("Select target column", columns)
    predictor_columns = st.multiselect("Select predictor columns", columns, default=[col for col in columns if col != target_column])
    
    if len(predictor_columns) > 0:

        predictors_col = customerloan[predictor_columns]
        target_col = customerloan[target_column].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(predictors_col, target_col, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = Sequential([
            Input(shape=(X_train.shape[1], )),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation=None, name="output_layer")
        ])

        model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["mae"])

        log_dir = "logs/" + datetime.datetime.now().strftime("%d_%m_%Y - %H_%M_%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        if st.button("Train Model"):
            history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_callback])

            st.subheader("Training Loss Plot")
            plt.figure(figsize=(8, 5))
            plt.plot(history.history["loss"], label="Training Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Mean Squared Error")
            plt.title("Training Loss vs Epochs")
            plt.legend()
            st.pyplot()

            y_pred = model.predict(X_test)

            st.subheader("Actual vs Predicted Values")
            plt.figure(figsize=(8, 5))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Actual vs Predicted")
            plt.legend()
            st.pyplot()

    else:
        st.warning("Please select at least one predictor column.")
else:
    st.warning("Please upload a CSV file to get started.")
