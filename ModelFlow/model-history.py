import streamlit, tensorflow, os, pandas, numpy, seaborn, pickle
from keras.api.models import Sequential, save_model, load_model
from keras.api.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from keras.api.callbacks import TensorBoard, LambdaCallback
from scipy import stats
from statsmodels import api
import datetime
from sklearn.impute import KNNImputer
from io import StringIO
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

save_path_root = "ModelFlow"
keras_model_save_path = os.path.join(save_path_root, "models", "saved-neural-network-Keras")
os.makedirs(keras_model_save_path, exist_ok=True)
saved_keras_model = [files for files in os.listdir(keras_model_save_path) if files.endswith(".h5")]
keras_model_choices = [""] + saved_keras_model
selected_keras_model = streamlit.selectbox("Neural Network Models", keras_model_choices, key="keras models")


@streamlit.cache_resource
def load_keras_model(keras_model_path):
    try:
        loaded_model = load_model(keras_model_path)
        return loaded_model
    except Exception as e:
        streamlit.error(f"Error loading Keras model: {e}")
        return None


def load_column_names(filepath):
    try:
        with open(filepath, 'r') as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        streamlit.error(f"Error: Column name file not found at {filepath}")
        return None
    except Exception as e:
        streamlit.error(f"Error reading column name file: {e}")
        return None


if selected_keras_model == "":
    streamlit.session_state["disable"] = True
    if keras_model_choices is None:
        streamlit.info("You haven’t saved any keras model yet.")
    else:
        streamlit.info("Please select a keras model to continue.")
else:
    keras_model_path = os.path.join(keras_model_save_path, selected_keras_model)
    keras_model_predictor_columns_path = os.path.join(keras_model_save_path, f"{os.path.splitext(selected_keras_model)[0]}.txt")
    keras_model_target_col_path = os.path.join(keras_model_save_path, f"{os.path.splitext(selected_keras_model)[0]}_target.txt")

    try:
        if selected_keras_model.endswith(".h5"):
            model = load_keras_model(keras_model_path)
            if model:
                predictor_columns = load_column_names(keras_model_predictor_columns_path) 
                target_col = load_column_names(keras_model_target_col_path)
                if predictor_columns and target_col:
                    input_values = {}
                    for col in predictor_columns:
                        input_values[col] = streamlit.number_input(f"{col}", key=f"input_{col}")

                    processed_input = []
                    for col in predictor_columns:
                        value = input_values[col]
                        if isinstance(value, (int, float)):
                            if value == 0:
                                processed_input.append(False)
                            elif value == 1:
                                processed_input.append(True)
                            else:
                                processed_input.append(float(value))
                        else:
                            processed_input.append(value)
                    input_array = numpy.array(processed_input).reshape(1, -1)

                    if streamlit.button(f"Predict {target_col[0]}", key="predict_button"): 
                        try:
                            prediction = model.predict(input_array)[0]
                            rounded_prediction = numpy.round(prediction, 2)
                            streamlit.write(f"Predicted {target_col[0]}: {rounded_prediction}") 
                        except Exception as e:
                            streamlit.error(f"Error during prediction: {e}")
                else:
                    streamlit.error("Could not load predictor or target column names.  Make sure they exist.")
            else:
                streamlit.error(f"model {selected_keras_model} failed to load")
        else:
            streamlit.warning("Selected file is not a Keras model (.h5).")
    except Exception as e:
        streamlit.error(f"Error : {e}")
