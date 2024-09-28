import streamlit as st
from backend.pdf_processing import extract_text_from_pdf
from backend.text_preprocessing import  split_into_chunks
from backend.question_answering import answer_question

# Streamlit App

# Title
st.title("PDF Question Answering System")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Question input text area
questions = st.text_area("Enter your questions (one per line)")

# Process the form when submit button is pressed
if uploaded_file and questions:
    # Display a progress bar
    with st.spinner("Processing the PDF..."):
        # Save the uploaded file temporarily
        pdf_filepath = "temp.pdf"
        with open(pdf_filepath, "wb") as f:
            f.write(uploaded_file.read())
        
        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(pdf_filepath)
        # cleaned_text = clean_text(pdf_text)
        chunks = split_into_chunks(pdf_text)

    # Split the questions by line
    # question_list = questions.splitlines()
    # Display answers
    st.header("Answers")
    answer = answer_question(questions,chunks)
    st.subheader(f"Question: {questions}")
    st.write(f"Answer: {answer.strip()}")
else:
    st.info("Please upload a PDF and enter your questions.")

if __name__ == '__main__':
    # Since this is a Streamlit app, Streamlit will handle the app running process.
    # No additional logic is needed here.
    pass