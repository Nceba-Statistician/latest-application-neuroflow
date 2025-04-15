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
streamlit.markdown("""<div class="titlemodel">Neural Network Model Builder for Prediction</div>""", unsafe_allow_html=True)

streamlit.subheader("")
Action_options = ["Select an action", "Select model fields", "Transform field values", "Update field data types",
                  "Determine Statistical Distribution"]
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
                            streamlit.success(f"✅ Field updated successfully!")
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

elif selected_action_option == "Determine Statistical Distribution":
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
                    if streamlit.checkbox("Preview distribution type"):
                        records_col = records[column_to_dist]
                        streamlit.write(f"Analyzing distribution of '{records_col.name}':")
                        skewness = stats.skew(records_col)
                        if skewness == 0.0:
                            streamlit.info(f"Skewness: {skewness:.3f} → perfectly symmetric")
                        elif -0.5 <= skewness <= 0.5:
                            streamlit.info(f"Skewness: {skewness:.3f} → approximately symmetric")  
                        elif -1.0 <= skewness < -0.5:  
                            streamlit.info(f"Skewness: {skewness:.3f} → longer tail on the left")
                        elif skewness < -1.0:
                            streamlit.info(f"Skewness: {skewness:.3f} → extreme longer tail on the left")
                        elif 0.5 < skewness <= 1.0:
                            streamlit.info(f"Skewness: {skewness:.3f} → longer tail on the right")  
                        elif skewness > 1.0:
                            streamlit.info(f"Skewness: {skewness:.3f} → extreme longer tail on the right")    

                        Kurtosis = stats.skew(records_col)
                        if Kurtosis == 0.0:
                            streamlit.info(f"Kurtosis: {Kurtosis:.3f} → normal distributed (mesokurtic)")
                        elif -0.5 <= Kurtosis <= 0.5:
                            streamlit.info(f"Kurtosis: {Kurtosis:.3f} → approximately normal")
                        elif -3.0 < Kurtosis < -0.5:
                            streamlit.info(f"Kurtosis: {Kurtosis:.3f} → light tailed, flatter peak (platykurtic) → fewer outliers") 
                        elif Kurtosis <= -3.0:
                            streamlit.info(f"Kurtosis: {Kurtosis:.3f} → Very flat, light tail (platykurtic) — fewer extreme outliers")                                  
                        elif 0.5 < Kurtosis < 3.0:  
                            streamlit.info(f"Kurtosis: {Kurtosis:.3f} → heavy tailed, sharp peak (leptokurtic) → more outliers")   
                        elif Kurtosis >= 3:
                            streamlit.info(f"Kurtosis: {Kurtosis:.3f} → Very peaked, fat tail (leptokurtic) → more extreme outliers")

                        streamlit.write("Statistical tests:")
                        shapiro_stat, shapiro_p = stats.shapiro(records_col)
                        dagostino_stat, dagostino_p = stats.normaltest(records_col)
                        anderson_result = stats.anderson(records_col, dist='norm')
                        streamlit.info(f"Shapiro-Wilk → Statistic: {shapiro_stat:.4f} and p-value: {shapiro_p:.4f}")
                        streamlit.info(f"D'Agostino K-squared → Statistic: {dagostino_stat:.4f} and p-value: {dagostino_p:.4f}")
                        streamlit.write("Anderson-Darling Test:")
                        streamlit.info(f"Test Statistic: {anderson_result.statistic:.4f}")
                        streamlit.info(f"Critical Values: {anderson_result.critical_values}")
                        streamlit.info(f"Significance Levels: {anderson_result.significance_level}")

                        # streamlit.info(f"Shapiro-Wilk | p-value: {stats.shapiro(records_col):.4f} | {stats.shapiro(records_col).pvalue:.4f}")
                        # streamlit.info(f"D'Agostino K-squared | p-value: {stats.normaltest(records_col):.4f} | {stats.normaltest(records_col).pvalue:.4f}")
                        # streamlit.info(f"Anderson-Darling statistic: {stats.anderson(records_col, dist='norm'):.4f}")
                        streamlit.write("The chosen significance level (α) is set to 0.05:") # 
                        streamlit.write("null hypothesis: data follow a normal distribution")
                        streamlit.write("alternative hypothesis: data do not follow a normal distribution")
                        alpha = 0.05
                        if stats.shapiro(records_col).pvalue <= alpha or stats.normaltest(records_col) <= alpha:
                            streamlit.success(f"✅ Shapiro-Wilk and D'Agostino K-squared statistical tests suggests your {records_col.name} data is likely not normally distributed (reject the null hypothesis)")
                        else:
                            streamlit.success(f"✅ Shapiro-Wilk and D'Agostino K-squared statistical tests suggests your {records_col.name} data is likely normally distributed (fail to reject the null hypothesis)")
                  
                        streamlit.info("If Shapiro-Wilk or D'Agostino K-squared is less than or equal to alpha (α), then reject your null hypothesis!")
                        
                        streamlit.write("Interpretation by comparing the test statistic to critical values")   
                        alpha_levels = [15, 10, 5, 2.5, 1] # Significance levels in percent
                        streamlit.write(f"Significance levels in percent: {alpha_levels}")
                        for i in range(len(anderson_result.critical_values)):
                             if anderson_result.statistic > anderson_result.critical_values[i]:
                                 streamlit.info(f"At the {alpha_levels[i]}% significance level, the test statistic ({anderson_result.statistic:.3f}) is greater than the critical value ({anderson_result.critical_values[i]:.3f}).")
                                 streamlit.info("Suggesting the data is likely not normally distributed (reject the null hypothesis).")
                                 break
                        else:
                            streamlit.info(f"The test statistic ({anderson_result.statistic:.3f}) is less than all critical values.")
                            streamlit.info(f"Suggesting the {records_col.name} data is likely normally distributed (fail to reject the null hypothesis at common alpha levels).")
                        streamlit.write("")
                        pyplot.figure(figsize=(5, 2))
                        # pyplot.subplot(1, 2, 1)
                        seaborn.histplot(records_col, kde=True, bins=30)
                        pyplot.title(f"Histogram with KDE - {records_col.name}") # Kernel Density Estimation will make our PDF smooth and continuous estimate 
                        pyplot.tight_layout()
                        streamlit.pyplot(pyplot) 
                        
                        pyplot.figure(figsize=(5, 2))
                        # pyplot.subplot(1, 2, 2)
                        api.qqplot(records_col, line="s")
                        pyplot.title(f"Q-Q Plot - {records_col.name}")
                        pyplot.tight_layout()
                        streamlit.pyplot(pyplot)                            

        except Exception as e:
            streamlit.error(f"Failed to load file: {e}")          

# 




# To add data value range an example 0 - 10 group it as "0 to 10"                   
