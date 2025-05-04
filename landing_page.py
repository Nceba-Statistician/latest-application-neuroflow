import streamlit, pandas

streamlit.write("Nceba Gagaza")
streamlit.info("Profile coming soon!")
streamlit.write("Education history, work experience, projects, etc")
CV_pdf = pandas.DataFrame([{
  "CV": "Nceba Gagaza"
}])
if streamlit.download_button("Download CV", data=CV_pdf, file_name="CV.txt"):
    streamlit.write("To be added.")
