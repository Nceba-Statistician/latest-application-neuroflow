import streamlit, tensorflow, os, pandas, numpy, seaborn, pickle
from keras.api.models import Sequential, save_model, load_model
from keras.api.layers import  Dense, Dropout, Input
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

streamlit.markdown("""<style> .font {font-size: 5px; font-weight: bold; background-color: green} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown("""<style> .titlemodel { position: absolute; font-size: 20px; left: 10px; top: 10px} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .footer { position: fixed; bottom: 0; left: 0; width: 20%; background-color: rgba(0, 0, 0, 0.05); padding: 10px; font-size: 12px; text-align: center;} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .emotion-cache {vertical-align: middle; overflow: hidden; color: inherit; fill: currentcolor; display: inline-flex; -webkit-box-align: center; align-items: center; font-size: 1.25rem; width: 1.25rem; height: 1.25rem; flex-shrink: 0; } </style>""",
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .weak-text {vertical-align: middle; position: absolute; font-size: 20px; color: gray; font-weight: bold;} </style>""",
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .simple-text {vertical-align: middle; position: absolute; font-size: 20px; color: gray;} </style>""",
    unsafe_allow_html=True
)
streamlit.markdown("""<div class="titlemodel">Neural Network Model Builder for Prediction</div>""", unsafe_allow_html=True)

streamlit.subheader("")
streamlit.info("Step-by-step guide to building models using traditional statistical methods and neural networks")
if streamlit.checkbox("Read Guide"):
    streamlit.write("")
    streamlit.markdown("<p class='weak-text'>Actions - skip steps when necessary</p>", unsafe_allow_html=True)
    streamlit.write("")
    col1_guide, col2_guide = streamlit.columns(2)
    with col1_guide:
        streamlit.write("Select model fields")
        with streamlit.expander("Purpose"):
            streamlit.write("Choose the key features needed for training and prediction, and assign them to a designated variable.")
        streamlit.write("Update field data types")
        with streamlit.expander("Purpose"):
            streamlit.write("Ensure each column has the correct data type (e.g., float for continuous variables, int for categories) to avoid model errors.")  
    with col2_guide:        
        streamlit.write("Transform field values")   
        with streamlit.expander("Purpose"):
            streamlit.write("Convert categorical values into numerical or standardized format for ML compatibility.")
        streamlit.write("Determine statistical distribution")
        with streamlit.expander("Purpose"):
            streamlit.write("Analyze the distribution of numeric features (e.g., skewed, Kurtosis, Gaussian, Logistic, Lognormal, Gumbel, Exponential, Weibull etc) to decide on transformations or statistical assumptions.") 

Action_options = ["", "Select model fields", "Transform field values", "Update field data types",
                  "Imputation", "Correlation Matrix", "Determine statistical distribution", "Model builder"]
selected_action_option = streamlit.selectbox("Choose an action:", Action_options, key="selectbox_action")

if selected_action_option == "":
    streamlit.session_state["disable"] = True
    streamlit.info("Please select an action to continue.")
elif selected_action_option == "Select model fields": 
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = [""] + saved_files
    selected_file = streamlit.selectbox("📂 Select file to adjust fields:", file_choices, key="selectbox_gen")
    if selected_file == "":
        streamlit.session_state["disable"] = True
        streamlit.info("Please select file to continue.")
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_object_action"):
                streamlit.write(records.head())          

            columns = records.columns.tolist()
            select_columns = streamlit.multiselect(
                "Select columns for your model", columns
                )
            if select_columns:
                selected_records = records[select_columns]
                if selected_records is not None:
                    streamlit.success(
                        "Done selecting? Continue selecting fields."
                        )
                file_name_input = streamlit.text_input("Enter a file name (without extension):", "")
                
                if streamlit.button(f"Store fields {file_name_input}"):
                    if not file_name_input.strip():
                        streamlit.warning("Please enter a valid file name.")
                    else:    
                        save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                        os.makedirs(save_path, exist_ok=True)
                        if selected_file.endswith(".csv"):
                            full_path = os.path.join(save_path, os.path.splitext(file_name_input)[0] + ".csv")
                            selected_records.to_csv(full_path, index=False)
                            streamlit.success(f"✅ {file_name_input} successfully saved! You can find file at manage-files.")
                        elif selected_file.endswith(".xlsx"):
                            full_path = os.path.join(save_path, os.path.splitext(file_name_input)[0] + ".xlsx")
                            selected_records.to_excel(full_path, index=False, engine='openpyxl')
                            streamlit.success(f"✅ {file_name_input} successfully saved! You can find file at manage-files.")
                 

        except Exception as e:
                streamlit.error(f"Failed to load file: {e}")

elif selected_action_option == "Transform field values":
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = [""] + saved_files
    selected_file = streamlit.selectbox("📂 Select file to transform:", file_choices, key="selectbox_trans")
    if selected_file == "":
        streamlit.session_state["disable"] = True
        streamlit.info("Please select file to continue.")
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_transform__object_before"):
                streamlit.write(records.head())   
            columns = records.columns.tolist()
            select_columns_trans = streamlit.multiselect(
                "Choose fileds you want to transform", columns
                )
            if select_columns_trans:
                column_to_map = streamlit.selectbox("Choose a column to transform values in", select_columns_trans, key="selectbox_map_column")
                if column_to_map:
                    streamlit.info("Define value mappings (e.g. Male → 1, Female → 0)")
                    if "value_map" not in streamlit.session_state:
                        streamlit.session_state["value_map"] = {}
                    with streamlit.form(key="mapping_form"):
                        new_key = streamlit.text_input("Enter original value (e.g. Male)", key="map_key")
                        new_value = streamlit.text_input("Enter new value (e.g. 1)", key="map_value")
                        submitted = streamlit.form_submit_button("➕ Add mapping")
                        
                        if submitted:
                            if new_key.strip() and new_value.strip():
                                streamlit.session_state["value_map"][new_key] = new_value
                                streamlit.success(f"Added: {new_key} → {new_value}")
                            else:
                                streamlit.warning("Please fill the fields!") 

                    if streamlit.session_state["value_map"]:
                        streamlit.write("Current mappings:") 
                        streamlit.json(streamlit.session_state["value_map"])
                        if streamlit.button(f"Apply mapping to {column_to_map}"):
                            records[f"{column_to_map}"] = records[f"{column_to_map}"].replace(streamlit.session_state["value_map"])
                            save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                            os.makedirs(save_path, exist_ok=True)
                            if selected_file.endswith(".csv"):
                                full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                                records.to_csv(full_path, index=False)
                                streamlit.success(f"✅ {column_to_map} updated successfully!")
                            elif selected_file.endswith(".xlsx"):
                                full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                                records.to_excel(full_path, index=False, engine='openpyxl') 
                                streamlit.success(f"✅ {column_to_map} updated successfully!") 
                        if streamlit.checkbox(f"Preview updated {selected_file}", key=f"preview_{selected_file}_transform_object_after"):
                            streamlit.write(records.head())
        except Exception as e:
            streamlit.error(f"Failed to load file: {e}") 

elif selected_action_option == "Update field data types":
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = [""] + saved_files
    selected_file = streamlit.selectbox("📂 Select file to update data types:", file_choices, key="selectbox_trans")
    if selected_file == "":
        streamlit.session_state["disable"] = True
        streamlit.info("Please select file to continue.")
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_dtypes_object"):
                streamlit.write(records.head())
            if streamlit.checkbox(f"Preview {selected_file} data types", key=f"preview_{selected_file}_dtypes_before"):
                streamlit.write(records.dtypes.reset_index().rename(columns={"index": "Fields", 0: "Data Type"})) 

            streamlit.info("Data types look good? Skip this step. Otherwise, update them below!")
            columns = records.columns.tolist()
            select_columns_dtypes = streamlit.multiselect(
                "Choose fileds you want to change data types", columns
                )
            if select_columns_dtypes:
                column_to_update = streamlit.selectbox("Choose a column to change data type in", select_columns_dtypes, key="selectbox_dtype_column")
                if column_to_update:
                    data_type_options = ["select data type", "int", "float", "str", "bool", "Datetime", "Time", "Date"]
                    selected_data_type = streamlit.selectbox("Choose data type", data_type_options, key="data_type_options_key")
                    try:
                        if selected_data_type == "select data type":
                            streamlit.session_state["disable"] = True
                            streamlit.info("Select data type to continue")
                        elif selected_data_type == "int":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                     records[f"{column_to_update}"] = pandas.to_numeric(records[f"{column_to_update}"], errors="raise").astype("int")
                                     save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                     os.makedirs(save_path, exist_ok=True)
                                     if selected_file.endswith(".csv"):
                                         full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                                         records.to_csv(full_path, index=False)
                                         streamlit.success(f"✅ {column_to_update} updated successfully!")
                                     elif selected_file.endswith(".xlsx"):
                                         full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                                         records.to_excel(full_path, index=False, engine='openpyxl')         
                                         streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "float":
                                precision = streamlit.number_input("Number of decimal places to round to:", min_value=0, step=1, value=2)
                                streamlit.write(f"Precision selected: {precision}")
                                if precision:
                                    if streamlit.button(f"Apply data type to {column_to_update}"):
                                        records[f"{column_to_update}"] = records[f"{column_to_update}"].astype("float").round(precision)
                                        save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                        os.makedirs(save_path, exist_ok=True)
                                        if selected_file.endswith(".csv"):
                                            full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                                            records.to_csv(full_path, index=False)
                                            streamlit.success(f"✅ {column_to_update} updated successfully!")
                                        elif selected_file.endswith(".xlsx"):
                                            full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                                            records.to_excel(full_path, index=False, engine='openpyxl')         
                                            streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "str":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                     records[f"{column_to_update}"] = records[f"{column_to_update}"].astype("str")
                                     save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                     os.makedirs(save_path, exist_ok=True)
                                     if selected_file.endswith(".csv"):
                                         full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                                         records.to_csv(full_path, index=False)
                                         streamlit.success(f"✅ {column_to_update} updated successfully!")
                                     elif selected_file.endswith(".xlsx"):
                                         full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                                         records.to_excel(full_path, index=False, engine='openpyxl')         
                                         streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "bool":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                     records[f"{column_to_update}"] = pandas.to_numeric(records[f"{column_to_update}"], errors="raise").astype("bool")
                                     save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                     os.makedirs(save_path, exist_ok=True)
                                     if selected_file.endswith(".csv"):
                                         full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                                         records.to_csv(full_path, index=False)
                                         streamlit.success(f"✅ {column_to_update} updated successfully!")
                                     elif selected_file.endswith(".xlsx"):
                                         full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                                         records.to_excel(full_path, index=False, engine='openpyxl')         
                                         streamlit.success(f"✅ {column_to_update} updated successfully!") 
                    # '2024-12-31 14:45'
                        elif selected_data_type == "Datetime":
                            if streamlit.button(f"Apply data type to {column_to_update}"):
                                records[f"{column_to_update}"] = pandas.to_datetime(records[f"{column_to_update}"], errors="raise")
                                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                os.makedirs(save_path, exist_ok=True)
                                if selected_file.endswith(".csv"):
                                    full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                                    records.to_csv(full_path, index=False)
                                    streamlit.success(f"✅ {column_to_update} updated successfully!")
                                elif selected_file.endswith(".xlsx"):
                                    full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                                    records.to_excel(full_path, index=False, engine='openpyxl')         
                                    streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "Time":
                            if streamlit.button(f"Apply data type to {column_to_update}"):
                                time_input = streamlit.text_input("Enter time in HH:MM:SS format", "")
                                if not time_input.strip():
                                    streamlit.warning("Please enter time")
                                else:
                                    records[f"{column_to_update}"] = pandas.to_datetime([time_input], format='%H:%M:%S').time[0]
                                    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                    os.makedirs(save_path, exist_ok=True)
                                    if selected_file.endswith(".csv"):
                                        full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                                        records.to_csv(full_path, index=False)
                                        streamlit.success(f"✅ {column_to_update} updated successfully!")
                                    elif selected_file.endswith(".xlsx"):
                                        full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                                        records.to_excel(full_path, index=False, engine='openpyxl') 
                                        streamlit.success(f"✅ {column_to_update} updated successfully!")
                        elif selected_data_type == "Date":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                    records[f"{column_to_update}"] = pandas.to_datetime(records[f"{column_to_update}"], errors="raise")
                                    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                    os.makedirs(save_path, exist_ok=True)
                                    if selected_file.endswith(".csv"):
                                        full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                                        records.to_csv(full_path, index=False)
                                        streamlit.success(f"✅ {column_to_update} updated successfully!")
                                    elif selected_file.endswith(".xlsx"):
                                        full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                                        records.to_excel(full_path, index=False, engine='openpyxl') 
                                        streamlit.success(f"✅ {column_to_update} updated successfully!")
                    except Exception as e:
                            streamlit.error(f"⚠️ Conversion failed: {e}")
                if streamlit.checkbox(f"Preview {selected_file} data types", key=f"preview_{selected_file}_dtypes_after"):
                    streamlit.write(records.dtypes.reset_index().rename(columns={"index": "Fields", 0: "Data Type"}))                        

        except Exception as e:
            streamlit.error(f"Failed to load file: {e}") 

elif selected_action_option == "Determine statistical distribution":
    with streamlit.expander("**Important**"):
        streamlit.write("Ensure you have dealt with your data's missing values!")
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = [""] + saved_files
    selected_file = streamlit.selectbox("📂 Select file to transform:", file_choices, key="selectbox_dist")
    if selected_file == "":
        streamlit.session_state["disable"] = True
        streamlit.info("Please select file to continue.")
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_transform__object_dist"):
                streamlit.write(records.head())
             
            records_columns = records.columns.tolist()
            select_columns_dist = streamlit.multiselect(
                "Choose fields to detect distribution", records_columns
                )
            if select_columns_dist:
                column_to_dist = streamlit.selectbox("Choose a column to continue", select_columns_dist, key="selectbox_dist_column")
                if column_to_dist:
                    streamlit.write("")
                    col1_distribution, col2_distribution = streamlit.columns(2)
                    with col1_distribution:
                        streamlit.write("Gaussian")
                        with streamlit.expander("Description"):
                            streamlit.write("A continuous probability distribution characterized by its bell-shaped curve. It is symmetrical around the mean, and its spread is determined by the standard deviation.")
                        streamlit.write("Lognormal")
                        with streamlit.expander("Description"):
                            streamlit.write("The logarithm of the variable is normally distributed. Often used for data that is positive and skewed.")

                        streamlit.write("Exponential")
                        with streamlit.expander("Description"):
                            streamlit.write("Describes the time between events in a Poisson process, where events occur continuously and independently at a constant average rate. Often used for modeling failure times.")

                    with col2_distribution:
                        streamlit.write("Logistic")
                        with streamlit.expander("Description"):
                            streamlit.write("S-shaped distribution similar to the normal distribution but with heavier tails. Its CDF is the sigmoid function, commonly used as an activation function in neural networks.")

                        streamlit.write("Gumbel")
                        with streamlit.expander("Description"):
                            streamlit.write("Used to model the distribution of the maximum (or minimum) of a number of independent, identically distributed random variables. Relevant in extreme value theory.")

                        streamlit.write("Weibull")
                        with streamlit.expander("Description"):
                            streamlit.write("A versatile distribution that can model a variety of shapes depending on its parameters. Used extensively in reliability analysis and survival analysis.")
                            
                    streamlit.write("")
                    streamlit.markdown("<p class='weak-text'>Preview distributions</p>", unsafe_allow_html=True)
                    streamlit.write("")
                    if streamlit.checkbox("Gaussian"):
                        records_col = records[column_to_dist]
                        
                        skewness = stats.skew(records_col)
                        def skewness_value_fun():
                            if skewness == 0.0:
                                return f"{skewness:.3f}"
                            elif -0.5 <= skewness <= 0.5:
                                return f"{skewness:.3f}"  
                            elif -1.0 <= skewness < -0.5:  
                                return f"{skewness:.3f}"
                            elif skewness < -1.0:
                                return f"{skewness:.3f}"
                            elif 0.5 < skewness <= 1.0:
                                return f"{skewness:.3f}" 
                            elif skewness > 1.0:
                                return f"{skewness:.3f}"
                        def skewness_description_fun():
                            if skewness == 0.0:
                                return "Perfectly symmetric"
                            elif -0.5 <= skewness <= 0.5:
                                return "Approximately symmetric" 
                            elif -1.0 <= skewness < -0.5:  
                                return "Longer tail on the left"
                            elif skewness < -1.0:
                                return "Extreme longer tail on the left"
                            elif 0.5 < skewness <= 1.0:
                                return "Longer tail on the right" 
                            elif skewness > 1.0:
                                return "Extreme longer tail on the right"
                                 
                        Kurtosis = stats.kurtosis(records_col)
                        
                        def Kurtosis_value_fun():
                            if Kurtosis == 0.0:
                                return f"{Kurtosis:.3f}"
                            elif -0.5 <= Kurtosis <= 0.5:
                                return f"{Kurtosis:.3f}"
                            elif -3.0 < Kurtosis < -0.5:
                                return f"{Kurtosis:.3f}"
                            elif Kurtosis <= -3.0:
                                return f"{Kurtosis:.3f}"                            
                            elif 0.5 < Kurtosis < 3.0:  
                                return f"{Kurtosis:.3f}"  
                            elif Kurtosis >= 3:
                                return f"{Kurtosis:.3f}"
                         
                        def Kurtosis_description_fun():
                            if Kurtosis == 0.0:
                                return "Normal distributed (mesokurtic)"
                            elif -0.5 <= Kurtosis <= 0.5:
                                return "Approximately normal"
                            elif -3.0 < Kurtosis < -0.5:
                                return "Light tailed, flatter peak (platykurtic) → fewer outliers"
                            elif Kurtosis <= -3.0:
                                return "Very flat, light tail (platykurtic) — fewer extreme outliers"                                
                            elif 0.5 < Kurtosis < 3.0:  
                                return "Heavy tailed, sharp peak (leptokurtic) → more outliers"  
                            elif Kurtosis >= 3:
                                return "Very peaked, fat tail (leptokurtic) → more extreme outliers"                        
                                                
                        Shapiro_Wilk, D_Agostino_K_squared, Anderson_Darling = streamlit.columns(3)
                        with Shapiro_Wilk:
                            streamlit.write("Shapiro-Wilk")
                            with streamlit.expander("Description"):
                                streamlit.write("The test statistic measures how well the data fits a normal distribution, with values close to 1 indicating normality.")
                        with D_Agostino_K_squared:
                            streamlit.write("D'Agostino K-squared")
                            with streamlit.expander("Description"):
                                streamlit.write("The test assesses normality by analyzing the skewness and kurtosis of the sample data. It calculates how far these sample moments deviate from the expected skewness (0) and kurtosis (3) of a normal distribution.") 
                        with Anderson_Darling:
                            streamlit.write("Anderson-Darling")
                            with streamlit.expander("Description"):
                                streamlit.write("It is a goodness-of-fit test that compares the cumulative distribution function (CDF) of your sample data to the CDF of the hypothesized distribution.") 

                        streamlit.write("")
                        
                        streamlit.write("Gaussian Hypothesis:")
                        
                        nullhyp, alterhyp = streamlit.columns(2)
                        with nullhyp:
                            streamlit.write("Null hypothesis")
                            with streamlit.expander("Assumption"):
                                streamlit.write("Data follow a normal distribution")
                        with alterhyp:
                            streamlit.write("Alternative hypothesis")
                            with streamlit.expander("Assumption"):
                                streamlit.write("Data do not follow a normal distribution")
                            
                        streamlit.write("")

                        streamlit.write(f"Calculated Measure for {records_col.name}:")                                                                                             
                        shape_dist = pandas.DataFrame({
                            "Measure": ["Skewness", "Kurtosis"],
                            "Values": [skewness_value_fun(), Kurtosis_value_fun()],
                            "Description": [skewness_description_fun(), Kurtosis_description_fun()]
                        })
                        streamlit.dataframe(shape_dist, hide_index=True) 
                        
                        shapiro_stat, shapiro_p = stats.shapiro(records_col)
                        dagostino_stat, dagostino_p = stats.normaltest(records_col)
                        anderson_result = stats.anderson(records_col, dist='norm')
                        alpha = 0.05
                        def Statistical_tests_conclusion_fun():
                            if shapiro_p <= alpha or dagostino_p <= alpha:
                                return f"✅ The p-values are less than/equal to the significance level (α = 0.05). This leads to the rejection of the null hypothesis of normality. Therefore, the data is likely not normally distributed."
                            else:
                                return f"✅ The p-values exceed the significance level (α = 0.05), indicating no significant deviation from normality. Thus, the data is likely normally distributed."
                        def Statistical_tests_shapiro_p_value_fun():
                            if shapiro_p <= alpha:
                                return f"{shapiro_p:.3f} <= {alpha}"
                            else:
                                return f"{shapiro_p:.3f} > {alpha}"
                        def Statistical_tests_dagostino_p_value_fun():
                            if dagostino_p <= alpha:
                                return f"{dagostino_p:.3f} <= {alpha}"
                            else:
                                return f"{dagostino_p:.3f} > {alpha}"
                        def Anderson_Darling_comp_stats_fun():
                            alpha_levels = [5, 2.5, 1] # Significance levels in percent
                            for i in range(len(anderson_result.critical_values)):
                                if anderson_result.statistic > anderson_result.critical_values[i]:
                                    return f"✅ At the {alpha_levels[i]}% significance level, the test statistic ({anderson_result.statistic:.3f}) exceeds the critical value ({anderson_result.critical_values[i]:.3f}). This leads to the rejection of the null hypothesis of normality. Therefore, the data is likely not normally distributed."
                                # break
                                elif anderson_result.statistic < anderson_result.critical_values[i]:
                                    return f"✅ At the {alpha_levels[i]}% significance level, the test statistic ({anderson_result.statistic:.3f}) is less than all critical values. Suggesting the {records_col.name} data is likely normally distributed (fail to reject the null hypothesis at common alpha levels)."

                        shapiro_stat, shapiro_p = stats.shapiro(records_col)
                        dagostino_stat, dagostino_p = stats.normaltest(records_col)
                        anderson_result = stats.anderson(records_col, dist='norm')
                        streamlit.write(f"Statistical tests for {records_col.name}")
                        
                        Anderson_Darling_stats = pandas.DataFrame({
                            "Statistical test": ["Anderson-Darling"],
                            "Statistic": [f"{anderson_result.statistic:.3f}"],
                            "Critical Values": [f"{anderson_result.critical_values}"],
                            "Significance Levels": [f"{anderson_result.significance_level}"]
                        })
                        streamlit.dataframe(Anderson_Darling_stats, hide_index=True)
                        streamlit.write("")
                        streamlit.markdown("<p class=weak-text>Interpretation by comparing the test statistic to critical values</>", unsafe_allow_html=True )
                        streamlit.write(f"{Anderson_Darling_comp_stats_fun()}")
                        streamlit.write("")
                        if numpy.isnan(dagostino_stat) or numpy.isnan(dagostino_p):
                            with streamlit.expander("Reason you getting 'nan'"):
                                streamlit.warning(
                                    f"You are getting 'nan' on 'D'Agostino K-squared Test' for **{records_col.name}** due to having Constant or Near-Constant Data "
                                    f"where if all or almost all of your data points have the same value, the variance will be zero (or very close to it). "
                                    f"This can lead to division by zero errors when calculating the test statistic, resulting in nan. "
                                    f"Another possibility is that your number of data points is very small (typically less than 20)."
                                    )
                        else:
                            ""
                        Statistical_tests = pandas.DataFrame({
                            "Statistical tests": ["Shapiro-Wilk", "D'Agostino K-squared"],
                            "Statistic": [f"{shapiro_stat:.3f}", f"{dagostino_stat:.3f}"],
                            "p-value": [f"{shapiro_p:.3f}", f"{dagostino_p:.3f}"],
                            "Alpha Comparison (α = 0.05)":[f"{Statistical_tests_shapiro_p_value_fun()}", f"{Statistical_tests_dagostino_p_value_fun()}"]
                        })
                        # Statistical_tests.set_index("Statistical tests", inplace=True)
                        streamlit.dataframe(
                            Statistical_tests, hide_index=True
                        )
                        streamlit.write("")
                        streamlit.markdown("<p class=weak-text>Conclution</>", unsafe_allow_html=True)
                        streamlit.write(f"{Statistical_tests_conclusion_fun()}") 
                        streamlit.write("")
                        
                        streamlit.write("")
                        pyplot.figure(figsize=(5, 2))
                        # pyplot.subplot(1, 2, 1)
                        seaborn.histplot(records_col, kde=True, bins=30)
                        pyplot.title(f"Histogram with KDE - {records_col.name}") # Kernel Density Estimation will make our PDF smooth and continuous estimate 
                        pyplot.tight_layout()
                        streamlit.pyplot(pyplot)                
                        
                        ppoints = numpy.linspace(0.01, 0.99, len(records_col))
                        quantiles_sample = numpy.quantile(records_col, ppoints)
                        quantiles_theoretical = stats.norm.ppf(ppoints)
                        
                        fig, ax = pyplot.subplots()
                        ax.scatter(quantiles_theoretical, quantiles_sample)
                        ax.plot([-4, 4], [-4, 4], color='r', linestyle='--')  # Line for perfect normality
                        ax.set_xlabel("Theoretical Quantiles (Standard Normal)")
                        ax.set_ylabel("Sample Quantiles")
                        ax.set_title("QQ Plot")
                        ax.grid(True)
                        streamlit.pyplot(fig)
                        
                        num_points = streamlit.slider("Number of data points:", min_value=10, max_value=500, value=100)
                        mean = streamlit.slider("Mean:", min_value=-5.0, max_value=5.0, value=0.0)
                        std_dev = streamlit.slider("Standard Deviation:", min_value=0.1, max_value=5.0, value=1.0)
                        data = numpy.random.normal(loc=mean, scale=std_dev, size=num_points)
                        ppoints = numpy.linspace(0.01, 0.99, len(records_col))
                        quantiles_sample = numpy.quantile(records_col, ppoints)
                        quantiles_theoretical = stats.norm.ppf(ppoints)
                        fig, ax = pyplot.subplots()
                        ax.scatter(quantiles_theoretical, quantiles_sample)
                        ax.plot([-4, 4], [-4, 4], color='r', linestyle='--')
                        ax.set_xlabel("Theoretical Quantiles (Standard Normal)")
                        ax.set_ylabel("Sample Quantiles")
                        ax.set_title("QQ Plot")
                        ax.grid(True)
                        streamlit.pyplot(fig)
                        
                        
                    elif streamlit.checkbox("Logistic"):
                        records_col = records[column_to_dist]
                        
                        skewness = stats.skew(records_col)
                        streamlit.write("Coming soon!")
                        
                    elif streamlit.checkbox("Lognormal"):
                        records_col = records[column_to_dist]
                        
                        skewness = stats.skew(records_col)
                        streamlit.write("Coming soon!")
                        
                    elif streamlit.checkbox("Gumbel"):
                        records_col = records[column_to_dist]
                        
                        skewness = stats.skew(records_col)
                        streamlit.write("Coming soon!")
                        
                    elif streamlit.checkbox("Exponential"):
                        records_col = records[column_to_dist]
                        
                        skewness = stats.skew(records_col)
                        streamlit.write("Coming soon!")
                        
                    elif streamlit.checkbox("Weibull"):
                        records_col = records[column_to_dist]
                        
                        skewness = stats.skew(records_col)
                        streamlit.write("Coming soon!")
                        # survival analysis with Weibull
                        

        except Exception as e:
            streamlit.error(f"Error could be on load file or selected field data types: {e}") 

elif selected_action_option == "Imputation":
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = [""] + saved_files
    selected_file = streamlit.selectbox("📂 Select file to impute:", file_choices, key="selectbox_impute")
    if selected_file == "":
        streamlit.session_state["disable"] = True
        streamlit.info("Please select file to continue.")
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_impute_object"):
                streamlit.write(records.head())
            if streamlit.checkbox(f"Check your {selected_file} nulls", key="selectbox_before_impute"):
                columns_with_missing = records.columns[records.isnull().any()].tolist()
                missing_counts_greater_than_zero = records.isnull().sum()[records.isnull().sum() > 0]
                buffer = StringIO()
                records.info(buf=buffer)
                columns_info = buffer.getvalue()
                col1_imp, col2_imp, col3_imp = streamlit.columns(3)
                with col1_imp:
                    if not columns_with_missing:
                        streamlit.write("columns with missing")
                        streamlit.info("No columns with missing values found.")
                    else:
                        streamlit.write("columns with missing")
                        streamlit.write(columns_with_missing) 
                with col2_imp:
                    streamlit.write("missing counts greater than zero")
                    if not missing_counts_greater_than_zero.empty:
                        missing_counts_greater_than_zero = missing_counts_greater_than_zero.rename("count")
                        streamlit.write(missing_counts_greater_than_zero)
                    else:
                        streamlit.info(f"No missing values found in the selected {selected_file}.")    
                with col3_imp:
                    streamlit.write("columns info")
                    streamlit.code(columns_info)      
             
            records_columns = records.columns.tolist()
            select_columns_dist = streamlit.multiselect(
                "Choose fields to impute", records_columns
                )
            if select_columns_dist:
                column_to_dist = streamlit.selectbox("Choose a column to continue", select_columns_dist, key="selectbox_impute_column")
                if column_to_dist:
                    streamlit.write("The 'K' in K-Nearest Neighbors (KNN) represents the number of nearest neighbors you want the algorithm to consider.")
                    with streamlit.expander("Warning"):
                        streamlit.write("When 'K' is too **small** (e.g., K=1 or K=2), the imputation for a missing value will be heavily influenced by only one or two very close neighbors." \
                        "If those neighbors happen to be outliers or contain noise, the imputed value will likely be inaccurate and not representative of the underlying data distribution." \
                        "This can introduce artificial variability and distort relationships in your data.")
                        streamlit.write("When 'K' is too **large**, the imputation for a missing value will be based on a larger and potentially more diverse set of neighbors." \
                        "This can smooth out local variations and potentially mask important patterns in the data.")

                    K = streamlit.number_input("Add KNNImputer", step=1, format="%d")
                    if K is not None and K > 0:
                        if streamlit.button(f"Impute and save your updated {selected_file}"):
                            try:
                                imputer = KNNImputer(n_neighbors = K, weights="uniform")
                                imputed_array = imputer.fit_transform(records[[column_to_dist]])
                                imputed_field = pandas.DataFrame(imputed_array, columns=[column_to_dist], index=records.index)
                                records[column_to_dist] = imputed_field
                                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                os.makedirs(save_path, exist_ok=True)
                                if selected_file.endswith(".csv"):
                                    full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                                    records.to_csv(full_path, index=False)
                                    streamlit.success(f"✅ You have successfully imputed and saved your {selected_file}!")
                                elif selected_file.endswith(".xlsx"):
                                    full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                                    records.to_excel(full_path, index=False, engine='openpyxl')
                                    streamlit.success(f"✅ You have successfully imputed and saved your {selected_file}!")
                            except Exception as e:
                                streamlit.error(f"An error occurred during imputation: {e}")  

                    else:
                        if K is not None and K <= 0:
                            streamlit.warning("Please enter a positive number of neighbors for KNNImputer.")
                    if streamlit.checkbox(f"Once imputed your {column_to_dist} check nulls", key="selectbox_after_impute"):
                        columns_with_missing = records.columns[records.isnull().any()].tolist()
                        missing_counts_greater_than_zero = records.isnull().sum()[records.isnull().sum() > 0]
                        buffer = StringIO()
                        records.info(buf=buffer)
                        columns_info = buffer.getvalue()
                        col1_imp, col2_imp, col3_imp = streamlit.columns(3)
                        with col1_imp:
                            if not columns_with_missing:
                                streamlit.write("columns with missing")
                                streamlit.info("No columns with missing values found.")
                            else:
                                streamlit.write("columns with missing")
                                streamlit.write(columns_with_missing) 
                        with col2_imp:
                            streamlit.write("missing counts greater than zero")
                            if not missing_counts_greater_than_zero.empty:
                                    missing_counts_greater_than_zero = missing_counts_greater_than_zero.rename("count")
                                    streamlit.write(missing_counts_greater_than_zero)
                            else:
                                streamlit.info(f"No missing values found in the selected {selected_file}.")    
                        with col3_imp:
                            streamlit.write("columns info")
                            streamlit.code(columns_info)          

        except Exception as e:
            streamlit.error(f"Failed to load file: {e}")

elif selected_action_option == "Correlation Matrix":
    with streamlit.expander("**Important**"):
        streamlit.write("Ensure you have dealt with your data's missing values and data types!")
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = [""] + saved_files
    selected_file = streamlit.selectbox("📂 Select file to Calculate Correlation Matrix:", file_choices, key="selectbox_dist")
    if selected_file == "":
        streamlit.session_state["disable"] = True
        streamlit.info("Please select file to continue.")
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_Correlation__object"):
                streamlit.write(records.head())
            numberOfPairwiseCorrelation = streamlit.selectbox("Select Pairwise Correlation type", ["", "Pairwise Correlation"])
            if numberOfPairwiseCorrelation == "":
                streamlit.info("Select Pairwise Correlation to continue")
            elif numberOfPairwiseCorrelation == "Pairwise Correlation":
                records_columns = records.columns.tolist()
                select_two_columns = streamlit.multiselect(
                    "Choose exactly two fields for specific Pairwise Correlation",
                    records_columns,
                    max_selections=2,
                    key="select_two_columns" 
                    )
                if len(select_two_columns) == 2:
                    first_feature = select_two_columns[0]
                    second_feature = select_two_columns[1]
                    if streamlit.checkbox(f"Show Correlation for {first_feature} and {second_feature}", key=f"show_corr_{first_feature}_{second_feature}"):
                        streamlit.write("Correlation Analysis")
                        try:
                            correlation_value = records[[first_feature, second_feature]].corr().iloc[0, 1]
                            streamlit.write(f"The correlation between {first_feature} and {second_feature} is: {correlation_value:.2f}")
                        except Exception as e:
                            streamlit.error(f"Error calculating correlation: {e}")
                elif select_two_columns:
                    streamlit.warning("Please select exactly two features for specific pairwise correlation.")
                else:
                    streamlit.info("Choose two fields to see their specific pairwise correlation.")

                streamlit.markdown("---")

                select_multiple_columns = streamlit.multiselect(
                    "Choose at least two fields for the Correlation Matrix",
                    records_columns,
                    key="select_multiple_columns" 
                    )
                if len(select_multiple_columns) >= 2:
                    if streamlit.checkbox("Show Correlation Matrix for selected fields", key="show_corr_matrix"):
                        streamlit.write("Correlation Matrix:")
                        try:
                            correlation_matrix_any = records[select_multiple_columns].corr()
                            streamlit.dataframe(correlation_matrix_any)
                            fig_matrix, ax_matrix = pyplot.subplots(figsize=(6, 4))
                            seaborn.heatmap(correlation_matrix_any, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_matrix)
                            streamlit.pyplot(fig_matrix)
                        except Exception as e:
                            streamlit.error(f"Error generating correlation matrix: {e}")
                elif select_multiple_columns:
                    streamlit.warning("Please select at least two features to display the correlation matrix.")
                else:
                    streamlit.info("Choose at least two fields to see their correlation matrix.")  
        except Exception as e:
            streamlit.error(f"Failed to load file: {e}")    

elif selected_action_option == "Model builder":
    with streamlit.expander("**Important**"):
        streamlit.write("Ensure you have dealt with your data's missing values and data types!")
    model_type_choise = streamlit.selectbox("Choose Model Approach",
                                            ["", "Traditional Statistical Methods", "Neural Networks"])
    if model_type_choise == "":
        streamlit.session_state["disable"] = True
    elif model_type_choise == "Traditional Statistical Methods":
        Distribution_Assumptions_choice = streamlit.selectbox("Choose Distribution Assumption",
                                            ["", "Distribution-Based Methods (Parametric)", "Distribution-Free Methods (Non-Parametric)"])
        if Distribution_Assumptions_choice == "":
            streamlit.session_state["disable"] = True
        elif Distribution_Assumptions_choice == "Distribution-Based Methods (Parametric)":
            save_path = os.path.join("ModelFlow", "data-config", "saved-files")
            os.makedirs(save_path, exist_ok=True)
            saved_files = [
                files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
                ]
            file_choices = [""] + saved_files
            selected_file = streamlit.selectbox("📂 Select file to build model:", file_choices, key="selectbox_model_builder_Parametric")
            if selected_file == "":
                streamlit.session_state["disable"] = True
                streamlit.info("Please select file to continue.")
            else:
                try:
                    file_path = os.path.join(save_path, selected_file)
                    if selected_file.endswith(".csv"):
                        records = pandas.read_csv(file_path)
                    elif selected_file.endswith(".xlsx"):
                        records = pandas.read_excel(file_path)
                    if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_model_builder_Parametric_object"):
                        streamlit.write(records.head())
                    Parametric_Methods_Focused_choice = streamlit.selectbox("Choose method",
                                                                            ["", "Methods Focused on Means and Differences of Means",
                                                                             "Methods Focused on Relationships Between Variables",
                                                                             "Methods Focused on Categorical Data (often relying on asymptotic normality)",
                                                                             "Other Parametric Methods"]
                                                                            )
                    if Parametric_Methods_Focused_choice == "":
                        streamlit.session_state["disable"] = True
                    elif Parametric_Methods_Focused_choice == "Methods Focused on Means and Differences of Means":
                        Means_and_Differences_of_Means_choice = streamlit.selectbox("Choose your specific method",
                                                                                    ["", "T-tests", "ANOVA", "ANCOVA"])
                        if Means_and_Differences_of_Means_choice == "":
                            streamlit.session_state["disable"] = True
                        elif Means_and_Differences_of_Means_choice == "T-tests":
                            with streamlit.expander("Student's t-tests"):
                                streamlit.write("One-sample t-test: Comparing a single sample mean to a known value")
                                streamlit.write("Independent samples t-test: Comparing means of two independent groups")
                                streamlit.write("Paired samples t-test: Comparing means of two related groups, like before-and-after measurements")
                        elif Means_and_Differences_of_Means_choice == "ANOVA":
                            with streamlit.expander("Analysis of Variance"):
                                streamlit.write("One-way ANOVA: Comparing means of three or more independent groups")
                                streamlit.write("Two-way ANOVA: Examining the effects of two independent variables on a dependent variable")
                                streamlit.write("Repeated measures ANOVA: Comparing means across multiple time points or conditions within the same subjects")
                        elif Means_and_Differences_of_Means_choice == "ANCOVA":
                            with streamlit.expander("Analysis of Covariance"):
                                streamlit.write("Extends ANOVA by including one or more covariates to control for extraneous variables")    
                    elif Parametric_Methods_Focused_choice == "Methods Focused on Relationships Between Variables":
                        Relationships_Between_Variables_choice = streamlit.selectbox("Choose your specific method",
                                                                                     ["", "Pearson Correlation",
                                                                                      "Linear Regression",
                                                                                      "Multiple Regression",
                                                                                      "Polynomial Regression"]
                                                                                     )
                        if Relationships_Between_Variables_choice == "":
                            streamlit.session_state["disable"] = True
                        elif Relationships_Between_Variables_choice == "Pearson Correlation":
                            with streamlit.expander("Description"):
                                streamlit.write("Measures the linear relationship between two continuous variables")
                        elif Relationships_Between_Variables_choice == "Linear Regression":
                            with streamlit.expander("Description"):
                                streamlit.write("Models the linear relationship between a dependent variable and one or more independent variables")
                        elif Relationships_Between_Variables_choice == "Multiple Regression":
                            with streamlit.expander("Description"):
                                streamlit.write("Extends linear regression to include multiple independent variables")
                        elif Relationships_Between_Variables_choice == "Polynomial Regression":
                            with streamlit.expander("Description"):
                                streamlit.write("Models non-linear relationships by including polynomial terms of the independent variable(s)")              
                    elif Parametric_Methods_Focused_choice == "Methods Focused on Categorical Data (often relying on asymptotic normality)":
                        Categorical_Data_choice = streamlit.selectbox("Choose your specific method", ["", "Chi-Square Tests"])
                        if Categorical_Data_choice == "":
                            streamlit.session_state["disable"] = True
                        elif Categorical_Data_choice == "Chi-Square Tests":
                            with streamlit.expander("Description"):
                                streamlit.write("Goodness-of-fit test: Comparing observed frequencies to expected frequencies")
                                streamlit.write("Test of independence: Examining the association between two categorical variables")
                                streamlit.write("Test of homogeneity: Comparing the distribution of a categorical variable across two or more groups")    
                    elif Parametric_Methods_Focused_choice == "Other Parametric Methods":
                        Other_Parametric_Methods = streamlit.selectbox("Choose your specific method",
                                                                       ["", "Z-tests", "Parametric Survival Analysis"]
                                                                       )
                        if Other_Parametric_Methods == "":
                            streamlit.session_state["disable"] = True
                        elif Other_Parametric_Methods == "Z-tests":
                            with streamlit.expander("Description"):
                                streamlit.write("Similar to t-tests but used when the population standard deviation is known or" \
                                "the sample size is large (relying on the Central Limit Theorem)")
                        elif Other_Parametric_Methods == "Parametric Survival Analysis":
                            with streamlit.expander("Description"):
                                streamlit.write("Methods like the exponential, Weibull, and log-normal models to analyze time-to-event data")                  
                except Exception as e:
                    streamlit.error(f"Failed to load file: {e}")

        elif Distribution_Assumptions_choice == "Distribution-Free Methods (Non-Parametric)":
            save_path = os.path.join("ModelFlow", "data-config", "saved-files")
            os.makedirs(save_path, exist_ok=True)
            saved_files = [
                files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
                ]
            file_choices = ["Select file to build model"] + saved_files
            selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_model_builder_Non_Parametric")
            if selected_file == "Select file to build model":
                streamlit.session_state["disable"] = True
                streamlit.info("Please select file to continue.")
            else:
                try:
                    file_path = os.path.join(save_path, selected_file)
                    if selected_file.endswith(".csv"):
                        records = pandas.read_csv(file_path)
                    elif selected_file.endswith(".xlsx"):
                        records = pandas.read_excel(file_path)
                    if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_model_builder_Non_Parametric_object"):
                        streamlit.write(records.head())
                except Exception as e:
                    streamlit.error(f"Failed to load file: {e}")

    elif model_type_choise == "Neural Networks":
        save_path = os.path.join("ModelFlow", "data-config", "saved-files")
        os.makedirs(save_path, exist_ok=True)
        saved_files = [
            files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
            ]
        file_choices = [""] + saved_files
        selected_file = streamlit.selectbox("📂 Select file to build model:", file_choices, key="selectbox_model_builder_NN")
        if selected_file == "":
            streamlit.session_state["disable"] = True
            streamlit.info("Please select file to continue.")
        else:
            try:
                file_path = os.path.join(save_path, selected_file)
                if selected_file.endswith(".csv"):
                    records = pandas.read_csv(file_path)
                elif selected_file.endswith(".xlsx"):
                    records = pandas.read_excel(file_path)
                if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_model_builder_NN_object"):
                    streamlit.write(records.head())

                NN_options = streamlit.selectbox("Select neural network architecture",
                                                 ["", "FNN – Feedforward Neural Network",
                                                  "CNN – Convolutional Neural Network",
                                                  "RNN – Recurrent Neural Network",
                                                  "LSTM – Long Short-Term Memory Network"]
                                                 )
                if NN_options == "": 
                    streamlit.session_state["disable"] = True
                elif NN_options == "FNN – Feedforward Neural Network":
                    columns_list = records.columns.tolist()
                    target_column = streamlit.selectbox("Select target column", [""] + columns_list, key="target")
                    predictor_columns = streamlit.multiselect("Select predictor columns", columns_list, key="predictors")
                    if len(predictor_columns) > 0 and target_column:
                        if target_column in predictor_columns:
                            streamlit.warning("Target column cannot be one of the predictor columns.")
                        else:
                            predictors_col = records[predictor_columns]
                            target_col = records[target_column].values.reshape(-1, 1)
                            test_size_percentage = streamlit.number_input(
                                "Add test size (e.g., 0.2)", min_value=0.0, max_value=1.0, step=0.1, format="%.1f"
                                )
                            First_dense_value = streamlit.number_input(
                                "Add First Dense (e.g., 64)", min_value=1, step=1, format="%d"
                                )
                            Second_dense_value = streamlit.number_input(
                                "Add second Dense (e.g., 32)", min_value=1, step=1, format="%d"
                                )
                            Dropout_value = streamlit.number_input(
                                "Add Dropout (e.g., 0.2)", min_value=0.0, max_value=1.0, step=0.1, format="%.1f"
                                ) # epochs
                            epochs_value = streamlit.number_input(
                                "Add epochs (e.g., 100)", min_value=10, step=1, format="%d"
                                )
                            Activation_function = streamlit.selectbox("Choose activation function", ["", "relu"])
                            Optional_date = streamlit.selectbox("Select date (Optional) for time steps", [""] + columns_list, key="time steps")
                            if test_size_percentage is not None and 0 < test_size_percentage <= 1 and First_dense_value > 0 and Second_dense_value > 0 and 0 <= Dropout_value < 1 and epochs_value >=10 and Activation_function:
                                X_train, X_test, y_train, y_test = train_test_split(predictors_col, target_col, test_size=test_size_percentage, random_state=42)                   
                                scaler = StandardScaler()
                                X_train = scaler.fit_transform(X_train)
                                X_test = scaler.transform(X_test)
                                
                                model = Sequential([
                                    Input(shape=(X_train.shape[1], )),
                                    Dense(First_dense_value, activation=Activation_function),
                                    Dropout(Dropout_value),
                                    Dense(Second_dense_value, activation=Activation_function),
                                    Dropout(Dropout_value),
                                    Dense(1, activation=None, name="output_layer")
                                    ])
                                
                                model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["mae"])
                                log_dir = "logs/" + datetime.datetime.now().strftime("%d_%m_%Y - %H_%M_%S")
                                tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

                                if streamlit.button("Train Model"):

                                    progress_bar = streamlit.progress(0)
                                    status_text = streamlit.empty()
                                    epochs = epochs_value
                                    
                                    def update_progress(epoch, logs):
                                        progress = (epoch + 1) / epochs
                                        progress_bar.progress(progress)
                                        status_text.text(f"Training in progress... Epoch {epoch + 1}/{epochs}")
                                        if epoch == epochs - 1:
                                            status_text.success("Training complete!")
                                            
                                    progress_callback = LambdaCallback(on_epoch_end=update_progress)    

                                    history = model.fit(
                                        X_train, y_train,
                                        epochs=epochs,
                                        callbacks=[tensorboard_callback, progress_callback],
                                        verbose=0  # Suppressed Keras output
                                        )

                                    y_pred = model.predict(X_test)

                                    mse = mean_squared_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)
                                    col_skew_kurt, col_JB, col_WD, col_CI, col_mse, col_r = streamlit.columns(6)
                                            
                                    with col_skew_kurt:
                                        with streamlit.expander("Skewness and Kurtosis"):
                                            streamlit.write("**Skewness**: Measure of the asymmetry of a probability distribution about its mean.")
                                            streamlit.write("**Kurtosis**: Measure of the 'tailedness'or peakedness of a probability distribution."
                                                            "It describes how heavy or light the tails of the distribution are relative to a normal distribution.")  
                                            streamlit.write("**Mesokurtic**: Kurtosis is around 3 (excess kurtosis around 0)")
                                            streamlit.write("**Leptokurtic**: Kurtosis is greater than 3 (excess kurtosis greater than 0)")
                                            streamlit.write("**Platykurtic**: Kurtosis is less than 3 (excess kurtosis less than 0)") 
                                            streamlit.write("**Since the kurtosis of a normal distribution is 3, then **Excess Kurtosis=Kurtosis−3**")

                                    with col_JB:
                                        with streamlit.expander("Jarque-Bera (JB)"):
                                            streamlit.write("is a goodness-of-fit test used to check if the residuals of a regression model follow a normal distribution." \
                                            "It assesses this by looking at the skewness (a measure of the asymmetry of the distribution)"
                                            "and kurtosis (a measure of the 'tailedness' or peakedness of the distribution) of the residuals.")
                                            streamlit.write("Null Hypothesis: The residuals are normally distributed.")
                                            streamlit.write("Alternative Hypothesis: The residuals are not normally distributed.")
                                            streamlit.write("If the residuals are not normally distributed, these inferences such as t-test and F-test might not be entirely reliable, especially in small samples.")

                                    with col_WD:
                                        with streamlit.expander("Durbin-Watson (DW)"):
                                            streamlit.write("Detect the presence of first-order autocorrelation in the residuals of a regression model. Autocorrelation means that the residuals are correlated with their own past values." \
                                                            "First-order autocorrelation specifically looks at the correlation between a residual and the residual immediately preceding it.")
                                            streamlit.write("Null Hypothesis: There is no first-order autocorrelation in the residuals.")
                                            streamlit.write("Alternative Hypothesis: There is first-order autocorrelation in the residuals (can be positive or negative).")
                                            streamlit.write("Interpretation: A DW statistic around **2** suggests little to no first-order autocorrelation.")
                                            streamlit.write("Interpretation: A value **significantly less than 2** suggests positive autocorrelation"
                                                            "(positive residuals tend to be followed by positive residuals, and negative by negative).")
                                            streamlit.write("Positive autocorrelation -> Example: If today's stock return was high, tomorrow's might also tend to be somewhat high.")
                                            streamlit.write("A value **significantly greater than 2** suggests negative autocorrelation"
                                                            "(positive residuals tend to be followed by negative residuals, and vice versa).")
                                            streamlit.write("Negative autocorrelation -> Example: In an assembly line, if one piece is slightly too long, the next might be made slightly shorter to compensate.")


                                    with col_CI:
                                        with streamlit.expander("Confidence Interval"):
                                            streamlit.write("If our value spans from a negative value to a positive value)," \
                                                            "it generally means that there is no statistically significant evidence at the chosen confidence level (e.g., 95%)" \
                                                                "that the true coefficient is different from zero.")
                                            streamlit.write("For uncertainty, the width of the interval reflects the uncertainty in our estimate of the coefficient." \
                                                            "A wider interval indicates more uncertainty, often due to factors like smaller sample size or higher variability in the data." \
                                                                "A narrower interval suggests a more precise estimate.")

                                        # metrics_description = pandas.DataFrame({
                                            # "Mean Square Error :": [f"{mse:.4f}"],
                                            # "R-Squared: ": [f"{r2:.4f}"]
                                           #  })
                                    # streamlit.markdown(metrics_description.style.hide(axis="index").to_html(), unsafe_allow_html=True)

                                    with col_mse:
                                        with streamlit.expander("Mean Square Error"):
                                            streamlit.write("Mean Square Error measures prediction accuracy. It tells us how far, on average, our model's predictions are from the true values. A lower MSE indicates better performance, with the metric being particularly sensitive to large prediction errors.")
                                    with col_r:
                                        with streamlit.expander("R-Squared"):
                                            streamlit.write("1 means the model perfectly explains the variance.")
                                            streamlit.write("0 means the model explains none of the variance.")       

                                    # lr = LinearRegression()
                                    # lr_FNN = lr.fit(X_train, y_train)

                                    X_train_constant = api.add_constant(X_train)
                                    model_FNN = api.OLS(y_train, X_train_constant).fit()
                                    streamlit.write(model_FNN.summary())

                                    fig, ax = pyplot.subplots(figsize=(6, 4))
                                    ax.plot(history.history["loss"], label="Training Loss")
                                    ax.set_xlabel("Epochs")
                                    ax.set_ylabel("Mean Squared Error")
                                    ax.set_title("Training Loss vs Epochs")
                                    ax.legend()
                                    streamlit.pyplot(fig)

                                    if Optional_date == "":
                                        observed_values = y_test.flatten()
                                        predicted_values = y_pred.flatten()
                                        time_steps = numpy.arange(len(observed_values))
                                        
                                        fig, ax = pyplot.subplots(figsize=(6, 4))
                                        ax.plot(time_steps, observed_values, label='Observed', marker='o', linestyle='-')
                                        ax.plot(time_steps, predicted_values, label='Predicted', marker='x', linestyle='--')
                                        ax.set_xlabel("Time Steps (or Index)")
                                        ax.set_ylabel("Value")
                                        ax.set_title("Observed vs. Predicted Values")
                                        ax.legend()
                                        ax.grid(True)
                                        streamlit.pyplot(fig)

                                    else:
                                        time_steps = records[Optional_date].values.flatten()[-len(y_test):]
                                        observed_values = y_test.flatten()
                                        predicted_values = y_pred.flatten()
                                        if len(time_steps) == len(observed_values):
                                            fig, ax = pyplot.subplots(figsize=(6, 4))
                                            ax.plot(time_steps, observed_values, label='Observed', marker='o', linestyle='-')
                                            ax.plot(time_steps, predicted_values, label='Predicted', marker='x', linestyle='--')
                                            ax.set_xlabel(Optional_date)
                                            ax.set_ylabel("Value")
                                            ax.set_title("Observed vs. Predicted Values Over Time")
                                            ax.legend()
                                            ax.grid(True)
                                            streamlit.pyplot(fig)
                                        else:
                                            streamlit.warning(f"The length of the selected date column ('{Optional_date}') does not match the length of the test data. Please check your data and date selection.")
                                if streamlit.checkbox("After training save your model"):
                                    streamlit.write("If you're satisfied with the model's performance, you may proceed to save it.")
                                    model_name = streamlit.text_input("Name your model")
                                    if model_name:
                                        if streamlit.button("Save model"):
                                            save_path_root = "ModelFlow"
                                            keras_model_save_path = os.path.join(save_path_root, "models", "saved-neural-network-Keras")
                                            os.makedirs(keras_model_save_path, exist_ok=True)
                                            full_path = os.path.join(keras_model_save_path, f"{model_name}.h5")
                                            try:
                                                save_model(model, full_path)
                                                streamlit.success(f"✅ You have successfully saved your Keras model '{model_name}'")
                                            except Exception as e:
                                                streamlit.error(f"An error occurred while saving the Keras model: {e}") 
                                            # try:
                                                # with open(full_path, "wb") as file:
                                                    # pickle.dump(model, file)
                                                # streamlit.success(f"✅ You have successfully saved your {model_name}!")    
                                            # except Exception as e:
                                                # streamlit.error(f"An error occurred while saving the model: {e}")
                                                # weights and biases: model.layers[i].get_weights()  
                                ""
                                save_path_root = "ModelFlow"
                                keras_model_save_path = os.path.join(save_path_root, "models", "saved-neural-network-Keras")
                                os.makedirs(keras_model_save_path, exist_ok=True)
                                saved_keras_model = [
                                    files for files in os.listdir(keras_model_save_path) if files.endswith(".png")
                                    ]
                                keras_model_choices = [""] + saved_keras_model
                                selected_keras_model = streamlit.selectbox("", keras_model_choices, key="keras models")
                                
                                @streamlit.cache_resource
                                def load_keras_model(model_path):
                                    try:
                                        loaded_model = load_model(model_path)
                                        return loaded_model
                                    except Exception as e:
                                        streamlit.error(f"Error loading Keras model: {e}")
                                        return None
                                if selected_keras_model == "":
                                    streamlit.session_state["disable"] = True
                                    if keras_model_choices is None:
                                        streamlit.info("You haven’t saved any keras model yet.")
                                    else:
                                        streamlit.info("Please select a keras model to continue.")
                                else:
                                    keras_model_path = os.path.join(keras_model_save_path, selected_keras_model)
                                    try:
                                        if selected_keras_model.endswith(".h5"):
                                            model = load_keras_model(keras_model_path)
                                            if model:
                                                input_values = {}
                                                for col in predictor_columns:
                                                    input_values[col] = streamlit.text_input(f"{col}", key=f"input_{col}")

                                                    if streamlit.button(f"Predict {target_column}"):
                                                        input_array = numpy.array([float(input_values[col]) for col in predictor_columns]).reshape(1, -1)
                                                        try:
                                                            prediction = model.predict(input_array)[0]
                                                            streamlit.write(f"Predicted {target_column}: {prediction}")
                                                        except Exception as e:
                                                            streamlit.error(f"Error during prediction: {e}")  
                                            else:
                                                streamlit.error(f"model {selected_keras_model} failed to load")    
                                            streamlit.write(f"Enter vales to predict {target_column}") # target_column predictor_columns
                                        else:
                                            streamlit.warning("Selected file is not a Keras model (.h5).")    
                                    except Exception as e:
                                        streamlit.error(f"Error : {e}") 

                            elif test_size_percentage is not None and (test_size_percentage <= 0 or test_size_percentage > 1):
                                streamlit.warning("Test size should be between 0.0 and 1.0 (exclusive of 0).")
                            elif First_dense_value <= 0 or Second_dense_value <= 0:
                                streamlit.warning("Dense layer values should be greater than 0.")
                            elif Dropout_value < 0 or Dropout_value >= 1:
                                streamlit.warning("Dropout value should be between 0.0 and less than 1.0.")
                            elif epochs_value < 10:
                                streamlit.warning("epochs should be at least 10 or above for better training process")
                                streamlit.write("Think of it like this:")
                                streamlit.write("You have a textbook (your training data).")
                                streamlit.write("One **epoch** is like reading the entire textbook from cover to cover once.")
                            elif not Activation_function:
                                streamlit.warning("Please choose an activation function.")

                    elif len(predictor_columns) < 0 and target_column:
                        streamlit.warning("Please select at least one predictor column.")



                elif NN_options == "CNN – Convolutional Neural Network":
                    "" 
                elif NN_options == "RNN – Recurrent Neural Network":
                    "" 
                elif NN_options == "LSTM – Long Short-Term Memory Network":
                    ""                                           

            except Exception as e:
                streamlit.error(f"Failed to load file: {e}")
                