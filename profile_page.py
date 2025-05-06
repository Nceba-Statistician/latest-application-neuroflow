import streamlit
from fpdf import FPDF
import base64

def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Nceba Gagaza", ln=True, align="C")
    pdf.cell(200, 10, txt="Profile coming soon!", ln=True, align="C")
    pdf.cell(200, 10, txt="Education history, work experience, specialization, etc", ln=True, align="C")
    return pdf.output(dest="S").encode("latin-1")

streamlit.write("Nceba Gagaza")
streamlit.info("Profile coming soon!")
streamlit.write("Education history, work experience, specialization, etc")

pdf_bytes = create_pdf()

streamlit.download_button(
    label="Download Profile (PDF)",
    data=pdf_bytes,
    file_name="Nceba_Gagaza_profile.pdf",
    mime="application/pdf",
)
