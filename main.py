import streamlit as st
from backend.pdf_processing import extract_text_from_pdf
from backend.text_preprocessing import split_into_chunks
from backend.question_answering import answer_question

# Streamlit App

# Title
st.title("PDF Question Answering System")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Question input text area inside a form
with st.form("question_form"):
    questions = st.text_area("Enter your questions (one per line)")
    submit_button = st.form_submit_button("Submit")

# Process the form when submit button is pressed
if submit_button and uploaded_file and questions:
    # Display a progress bar
    with st.spinner("Processing the PDF..."):
        # Save the uploaded file temporarily
        pdf_filepath = "temp.pdf"
        with open(pdf_filepath, "wb") as f:
            f.write(uploaded_file.read())
        
        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(pdf_filepath)
        chunks = split_into_chunks(pdf_text)

    # Split the questions by line
    questions_list = questions.splitlines()
    
    # Store questions and their answers in a dictionary
    qa = {}
    for question in questions_list:
        answer = answer_question(question, chunks)
        qa[question] = answer
    
    # Display the results
    st.header("Answers")
    for question, answer in qa.items():
        st.subheader(f"Question: {question}")
        st.write(f"Answer: {answer.strip()}")
else:
    if not submit_button:
        st.info("Please fill in the form and submit.")
    if not uploaded_file:
        st.info("Please upload a PDF.")
    if not questions:
        st.info("Please enter some questions.")

if __name__ == '__main__':
    pass
