import streamlit, pandas

streamlit.write("Nceba Gagaza")
streamlit.info("Profile coming soon!")
streamlit.write("Education history, work experience, projects, etc")
if streamlit.download_button("Download CV", data="", file_name="CV.txt"):
    streamlit.write("To be added.")
