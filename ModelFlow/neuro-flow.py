import streamlit, tensorflow, os, pandas, numpy, seaborn 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from tensorflow.python.keras.callbacks import TensorBoard
from scipy import stats
from statsmodels import api
import datetime
from sklearn.impute import KNNImputer
from io import StringIO


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

Action_options = ["Select an action", "Select model fields", "Transform field values", "Update field data types",
                  "Determine statistical distribution", "Imputation"]
selected_action_option = streamlit.selectbox("Choose an action:", Action_options, key="selectbox_action")

if selected_action_option == "Select an action":
    streamlit.session_state["disable"] = True
    streamlit.info("Please select an action to continue.")
elif selected_action_option == "Select model fields": 
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = ["Select file to adjust fields"] + saved_files
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_gen")
    if selected_file == "Select file to adjust fields":
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
                        full_path = os.path.join(save_path, f"{file_name_input}.csv")
                        selected_records.to_csv(full_path, index=False)
                        streamlit.success(f"✅ {file_name_input} successfully saved! You can find file at manage-files.")
        except Exception as e:
                streamlit.error(f"Failed to load file: {e}")

elif selected_action_option == "Transform field values":
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = ["Select file to transform"] + saved_files
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_trans")
    if selected_file == "Select file to transform":
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
                            full_path = os.path.join(save_path, selected_file)
                            records.to_csv(full_path, index=False)
                            streamlit.success(f"✅ {selected_file} updated successfully!")
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
    file_choices = ["Select file to update data types"] + saved_files
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_trans")
    if selected_file == "Select file to update data types":
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
                                     full_path = os.path.join(save_path, selected_file)
                                     records.to_csv(full_path, index=False)         
                                     streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "float":
                                precision = streamlit.slider("Select decimal precision", min_value=0, max_value=10, value=2)
                                streamlit.write(f"Precision selected: {precision}")
                                if precision:
                                    if streamlit.button(f"Apply data type to {column_to_update}"):
                                        records[f"{column_to_update}"] = pandas.to_numeric(records[f"{column_to_update}"], errors="raise").astype("float").round(precision)
                                        save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                        os.makedirs(save_path, exist_ok=True)
                                        full_path = os.path.join(save_path, selected_file)
                                        records.to_csv(full_path, index=False)         
                                        streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "str":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                     records[f"{column_to_update}"] = pandas.to_numeric(records[f"{column_to_update}"], errors="raise").astype("str")
                                     save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                     os.makedirs(save_path, exist_ok=True)
                                     full_path = os.path.join(save_path, selected_file)
                                     records.to_csv(full_path, index=False)         
                                     streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "bool":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                     records[f"{column_to_update}"] = pandas.to_numeric(records[f"{column_to_update}"], errors="raise").astype("bool")
                                     save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                     os.makedirs(save_path, exist_ok=True)
                                     full_path = os.path.join(save_path, selected_file)
                                     records.to_csv(full_path, index=False)         
                                     streamlit.success(f"✅ {column_to_update} updated successfully!") 
                    # '2024-12-31 14:45'
                        elif selected_data_type == "Datetime":
                            if streamlit.button(f"Apply data type to {column_to_update}"):
                                records[f"{column_to_update}"] = pandas.to_datetime(records[f"{column_to_update}"], errors="raise")
                                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                os.makedirs(save_path, exist_ok=True)
                                full_path = os.path.join(save_path, selected_file)
                                records.to_csv(full_path, index=False)         
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
                                    full_path = os.path.join(save_path, selected_file)
                                    records.to_csv(full_path, index=False)         
                                    streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "Date":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                    records[f"{column_to_update}"] = pandas.to_datetime(records[f"{column_to_update}"], errors="raise")
                                    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                    os.makedirs(save_path, exist_ok=True)
                                    full_path = os.path.join(save_path, selected_file)
                                    records.to_csv(full_path, index=False)         
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
    file_choices = ["Select file to transform"] + saved_files
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_dist")
    if selected_file == "Select file to transform":
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
            streamlit.error(f"Failed to load file: {e}") 

elif selected_action_option == "Imputation":
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = ["Select file to impute"] + saved_files
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_impute")
    if selected_file == "Select file to impute":
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
            if streamlit.checkbox(f"Preview your {selected_file}", key="selectbox_before_impute"):
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
                                full_path = os.path.join(save_path, selected_file)
                                records.to_csv(full_path, index=False)         
                                streamlit.success(f"✅ You have successfully imputed and saved your {selected_file}!")
                            except Exception as e:
                                streamlit.error(f"An error occurred during imputation: {e}")  

                    else:
                        if K is not None and K <= 0:
                            streamlit.warning("Please enter a positive number of neighbors for KNNImputer.")
                        if streamlit.checkbox("Preview your ", key="selectbox_after_impute"):
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


# To add data value range an example 0 - 10 group it as "0 to 10" 