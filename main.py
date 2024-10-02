import streamlit as st
import asyncio
from backend.pdf_processing import extract_text_from_pdf
from backend.text_preprocessing import split_into_chunks
from backend.question_answering import answer_question, initialize_openai_client, initialize_vectorstore
import time

# Streamlit App
st.title("PDF Question Answering System")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Question input text area inside a form
with st.form("question_form"):
    questions = st.text_area("Enter your questions (one per line)")
    submit_button = st.form_submit_button("Submit")

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
if 'original_chunks' not in st.session_state:
    st.session_state['original_chunks'] = None

async def process_pdf(pdf_file):
    """Process the uploaded PDF to extract text and create a vector store."""
    start_time = time.time()
    global vectorstore, original_chunks  # Use global variables to store the vector store and chunks
    pdf_filepath = "temp.pdf"
    with open(pdf_filepath, "wb") as f:
        f.write(pdf_file.read())

    # Extract text from the uploaded PDF
    pdf_text = await extract_text_from_pdf(pdf_filepath)
    chunks = split_into_chunks(pdf_text)

    # Initialize OpenAI client and create vector store
    client = await initialize_openai_client()
    vectorstore, original_chunks = initialize_vectorstore(chunks)

    st.session_state['vectorstore'] = vectorstore
    st.session_state['original_chunks'] = original_chunks

    st.success("PDF processed and vector store created.")
    print("--- %s pdf preprocessing time in seconds ---" % (time.time() - start_time))

async def fetch_answers(questions_list):
    """Fetch answers for the provided questions."""
    qa = {}
    client = await initialize_openai_client()  # Ensure OpenAI client is initialized

    tasks = []  # List to hold tasks for answers
    for question in questions_list:
        if st.session_state['vectorstore'] is not None and st.session_state['original_chunks'] is not None:  # Check if vector store is created
            tasks.append(answer_question(question, st.session_state['vectorstore'], st.session_state['original_chunks'], client))
        else:
            qa[question] = "Please upload a PDF first."
    
    # Gather results concurrently
    if tasks:  # Only gather if there are tasks
        start_time = time.time()
        answers = await asyncio.gather(*tasks)
        print("--- %s q ---" % (time.time() - start_time))
        
        # Populate the qa dictionary with results
        for question, answer in zip(questions_list, answers):
            qa[question] = answer
    
    return qa

# Process the PDF as soon as it's uploaded
if uploaded_file is not None and st.session_state['vectorstore'] is None:
    asyncio.run(process_pdf(uploaded_file))

# Process questions when the submit button is pressed
if submit_button and questions:
    questions_list = questions.splitlines()
    
    # Run the async process to fetch answers and display results
    qa = asyncio.run(fetch_answers(questions_list))

    # Display the results
    st.header("Answers")
    for question, answer in qa.items():
        st.subheader(f"Question: {question}")
        st.write(f"Answer: {answer}")

else:
    if not submit_button:
        st.info("Please fill in the form and submit your questions.")
    if not uploaded_file:
        st.info("Please upload a PDF.")
    if not questions:
        st.info("Please enter some questions.")

if __name__ == '__main__':
    pass
