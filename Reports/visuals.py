import streamlit, pandas, numpy, datetime, json, requests, pyodbc, os, seaborn
from matplotlib import pyplot

save_path = os.path.join("ModelFlow", "data-config", "saved-files")
os.makedirs(save_path, exist_ok=True)
saved_files = [
    files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
    ]
file_choices = ["Select file to view fields"] + saved_files

col1_dash, col2_dash, col3_dash, col5_dash = streamlit.columns(4)

with col5_dash:
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_dashboard", label_visibility="collapsed")
    if selected_file == "Select file to view fields":
        streamlit.session_state["disable"] = True
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            def load_data(file_path):
                if file_path.endswith(".csv"):
                    records = pandas.read_csv(file_path)
                    return records
                elif file_path.endswith(".xlsx"):
                    records = pandas.read_excel(file_path)
                    return records

            if streamlit.checkbox(f"📄 Fields {selected_file}", key=f"preview_{selected_file}_dashboard_object"):
                for column in load_data(file_path).columns:
                    streamlit.write(column)
        except Exception as e:
            streamlit.error(f"Failed to load file: {e}") 
                       
with col1_dash:
    all_records = None
    try:
        all_records = load_data(file_path)
        if all_records is not None:
            numeric_columns = all_records.select_dtypes(include=numpy.number).columns.tolist()
            all_columns = all_records.columns.tolist() 
            
            chart_type = streamlit.selectbox("Select Chart Type:", ["", "Scatter Plot", "Bar Chart", "Line Chart", "Pie Chart", "Stacked Bar Chart"], key=col1_dash)
            x_axis = streamlit.selectbox("Select X-axis:", [""] + all_columns, key=col1_dash)
            y_axis = streamlit.selectbox("Select Y-axis:", [""] + all_columns, key=col1_dash)
            color_option = streamlit.selectbox("Optional Color Field:", [None] + all_columns, key=col1_dash)
            if x_axis and y_axis:
                try:
                    fig, ax = pyplot.subplots(figsize=(6, 4))
                    if chart_type == "Scatter Plot":
                        if color_option:
                            seaborn.scatterplot(x=x_axis, y=y_axis, hue=color_option, data=all_records, ax=ax)
                        else:
                            ax.scatter(all_records[x_axis], all_records[y_axis])
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                    
                    elif chart_type == "Bar Chart":
                        if all_records[x_axis].dtype == "object" or all_records[y_axis].dtype == "object":
                            if color_option:
                                seaborn.barplot(x=x_axis, y=y_axis, hue=color_option, data=all_records, ax=ax)
                            else:
                                seaborn.barplot(x=x_axis, y=y_axis, data=all_records, ax=ax)
                                ax.set_xlabel(x_axis)
                                ax.set_ylabel(y_axis)
                        else:
                             streamlit.warning("For Bar Chart, at least one selected axis should be categorical (non-numeric).")
                           
                    elif chart_type == "Line Chart":
                        if color_option:
                            seaborn.lineplot(x=x_axis, y=y_axis, hue=color_option, data=all_records, ax=ax)
                        else:
                            ax.plot(all_records[x_axis], all_records[y_axis])
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            pyplot.xticks(rotation=45, ha="right")
                    elif chart_type == "Pie Chart":
                        streamlit.write(f"You are only using the {x_axis} column for this chart.")
                        if all_records[x_axis].dtype == "object":
                            value_counts = all_records[x_axis].value_counts()
                            ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=90)
                            ax.axis("equal")
                        else:
                            streamlit.warning("For Pie Chart, the X-axis should be a categorical column.") 
                            
                    elif chart_type == "Stacked Bar Chart":
                        if all_records[x_axis].dtype == "object" and all_records[y_axis].dtype != "object" and color_option and all_records[color_option].dtype == "object":
                            grouped_data = all_records.groupby([x_axis, color_option])[y_axis].sum().unstack()
                            grouped_data.plot(kind="bar", stacked=True, ax=ax)
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            ax.legend(title=color_option)
                            pyplot.xticks(rotation=45, ha="right")
                                         
                        else:
                            streamlit.warning("For Stacked Bar Chart, X-axis should be categorical, Y-axis should be numeric, and an optional categorical Color Field is required.")               
                    streamlit.pyplot(fig)
                    chart_name = streamlit.text_input("", placeholder="name this chart", key=col1_dash)
                    if streamlit.button(f"save {chart_name}", key=col1_dash):
                        if chart_name:
                            save_path_root = "Reports"
                            charts_save_path = os.path.join(save_path_root, "charts-config", "saved-charts")
                            os.makedirs(charts_save_path, exist_ok=True)
                            full_path = os.path.join(charts_save_path, f"{chart_name}.png")
                            fig.savefig(full_path, bbox_inches="tight")
                            streamlit.success(f"✅ You have successfully saved {chart_name}! - check saved charts at Manage files")
                        else:
                            streamlit.warning("Please enter a name for the chart before saving.")    
                        
                except Exception as e:
                        streamlit.error(f"Error generating chart: {e}") 
    except Exception as e:
        streamlit.info("Please load data for analysis")
with col2_dash:
    all_records = None
    try:
        all_records = load_data(file_path)
        if all_records is not None:
            numeric_columns = all_records.select_dtypes(include=numpy.number).columns.tolist()
            all_columns = all_records.columns.tolist() 
            
            chart_type = streamlit.selectbox("Select Chart Type:", ["", "Scatter Plot", "Bar Chart", "Line Chart", "Pie Chart", "Stacked Bar Chart"], key=col2_dash)
            x_axis = streamlit.selectbox("Select X-axis:", [""] + all_columns, key=col2_dash)
            y_axis = streamlit.selectbox("Select Y-axis:", [""] + all_columns, key=col2_dash)
            color_option = streamlit.selectbox("Optional Color Field:", [None] + all_columns, key=col2_dash)
            if x_axis and y_axis:
                try:
                    fig, ax = pyplot.subplots(figsize=(6, 4))
                    if chart_type == "Scatter Plot":
                        if color_option:
                            seaborn.scatterplot(x=x_axis, y=y_axis, hue=color_option, data=all_records, ax=ax)
                        else:
                            ax.scatter(all_records[x_axis], all_records[y_axis])
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                    
                    elif chart_type == "Bar Chart":
                        if all_records[x_axis].dtype == "object" or all_records[y_axis].dtype == "object":
                            if color_option:
                                seaborn.barplot(x=x_axis, y=y_axis, hue=color_option, data=all_records, ax=ax)
                            else:
                                seaborn.barplot(x=x_axis, y=y_axis, data=all_records, ax=ax)
                                ax.set_xlabel(x_axis)
                                ax.set_ylabel(y_axis)
                        else:
                             streamlit.warning("For Bar Chart, at least one selected axis should be categorical (non-numeric).")
                           
                    elif chart_type == "Line Chart":
                        if color_option:
                            seaborn.lineplot(x=x_axis, y=y_axis, hue=color_option, data=all_records, ax=ax)
                        else:
                            ax.plot(all_records[x_axis], all_records[y_axis])
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            pyplot.xticks(rotation=45, ha="right")
                    elif chart_type == "Pie Chart":
                        streamlit.write(f"You are only using the {x_axis} column for this chart.")
                        if all_records[x_axis].dtype == "object":
                            value_counts = all_records[x_axis].value_counts()
                            ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=90)
                            ax.axis("equal")
                        else:
                            streamlit.warning("For Pie Chart, the X-axis should be a categorical column.") 
                            
                    elif chart_type == "Stacked Bar Chart":
                        if all_records[x_axis].dtype == "object" and all_records[y_axis].dtype != "object" and color_option and all_records[color_option].dtype == "object":
                            grouped_data = all_records.groupby([x_axis, color_option])[y_axis].sum().unstack()
                            grouped_data.plot(kind="bar", stacked=True, ax=ax)
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            ax.legend(title=color_option)
                            pyplot.xticks(rotation=45, ha="right")
                                         
                        else:
                            streamlit.warning("For Stacked Bar Chart, X-axis should be categorical, Y-axis should be numeric, and an optional categorical Color Field is required.")               
                    streamlit.pyplot(fig) 
                    chart_name = streamlit.text_input("", placeholder="name this chart", key=col2_dash)
                    if streamlit.button(f"save {chart_name}", key=col2_dash):
                        if chart_name:
                            save_path_root = "Reports"
                            charts_save_path = os.path.join(save_path_root, "charts-config", "saved-charts")
                            os.makedirs(charts_save_path, exist_ok=True)
                            full_path = os.path.join(charts_save_path, f"{chart_name}.png")
                            fig.savefig(full_path, bbox_inches="tight")
                            streamlit.success(f"✅ You have successfully saved {chart_name}! - check saved charts at Manage files")                        
                except Exception as e:
                        streamlit.error(f"Error generating chart: {e}") 
    except Exception as e:
        streamlit.info("Please load data for analysis")   
with col3_dash:
    all_records = None
    try:
        all_records = load_data(file_path)
        if all_records is not None:
            numeric_columns = all_records.select_dtypes(include=numpy.number).columns.tolist()
            all_columns = all_records.columns.tolist() 
            
            chart_type = streamlit.selectbox("Select Chart Type:", ["", "Scatter Plot", "Bar Chart", "Line Chart", "Pie Chart", "Stacked Bar Chart"], key=col3_dash)
            x_axis = streamlit.selectbox("Select X-axis:", [""] + all_columns, key=col3_dash)
            y_axis = streamlit.selectbox("Select Y-axis:", [""] + all_columns, key=col3_dash)
            color_option = streamlit.selectbox("Optional Color Field:", [None] + all_columns, key=col3_dash)
            if x_axis and y_axis:
                try:
                    fig, ax = pyplot.subplots(figsize=(6, 4))
                    if chart_type == "Scatter Plot":
                        if color_option:
                            seaborn.scatterplot(x=x_axis, y=y_axis, hue=color_option, data=all_records, ax=ax)
                        else:
                            ax.scatter(all_records[x_axis], all_records[y_axis])
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                    
                    elif chart_type == "Bar Chart":
                        if all_records[x_axis].dtype == "object" or all_records[y_axis].dtype == "object":
                            if color_option:
                                seaborn.barplot(x=x_axis, y=y_axis, hue=color_option, data=all_records, ax=ax)
                            else:
                                seaborn.barplot(x=x_axis, y=y_axis, data=all_records, ax=ax)
                                ax.set_xlabel(x_axis)
                                ax.set_ylabel(y_axis)
                        else:
                             streamlit.warning("For Bar Chart, at least one selected axis should be categorical (non-numeric).")
                           
                    elif chart_type == "Line Chart":
                        if color_option:
                            seaborn.lineplot(x=x_axis, y=y_axis, hue=color_option, data=all_records, ax=ax)
                        else:
                            ax.plot(all_records[x_axis], all_records[y_axis])
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            pyplot.xticks(rotation=45, ha="right")
                    elif chart_type == "Pie Chart":
                        streamlit.write(f"You are only using the {x_axis} column for this chart.")
                        if all_records[x_axis].dtype == "object":
                            value_counts = all_records[x_axis].value_counts()
                            ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=90)
                            ax.axis("equal")
                        else:
                            streamlit.warning("For Pie Chart, the X-axis should be a categorical column.") 
                            
                    elif chart_type == "Stacked Bar Chart":
                        if all_records[x_axis].dtype == "object" and all_records[y_axis].dtype != "object" and color_option and all_records[color_option].dtype == "object":
                            grouped_data = all_records.groupby([x_axis, color_option])[y_axis].sum().unstack()
                            grouped_data.plot(kind="bar", stacked=True, ax=ax)
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            ax.legend(title=color_option)
                            pyplot.xticks(rotation=45, ha="right")
                                         
                        else:
                            streamlit.warning("For Stacked Bar Chart, X-axis should be categorical, Y-axis should be numeric, and an optional categorical Color Field is required.")               
                    streamlit.pyplot(fig)    
                    chart_name = streamlit.text_input("", placeholder="name this chart", key=col3_dash)
                    if streamlit.button(f"save {chart_name}", key=col3_dash):
                        if chart_name:
                            save_path_root = "Reports"
                            charts_save_path = os.path.join(save_path_root, "charts-config", "saved-charts")
                            os.makedirs(charts_save_path, exist_ok=True)
                            full_path = os.path.join(charts_save_path, f"{chart_name}.png")
                            fig.savefig(full_path, bbox_inches="tight")
                            streamlit.success(f"✅ You have successfully saved {chart_name}! - check saved charts at Manage files")                        
                except Exception as e:
                        streamlit.error(f"Error generating chart: {e}") 
    except Exception as e:
        streamlit.info("Please load data for analysis")  
