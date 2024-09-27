from fastapi import FastAPI, UploadFile, File
from .pdf_processing import extract_text_from_pdf
from .text_preprocessing import clean_text, split_into_chunks
from .question_answering import answer_question

app = FastAPI()

pdf_filepath = None 


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_filepath 
    
    # Check if a previous PDF exists and delete it
    if pdf_filepath:
        os.remove(pdf_filepath)
    
    # Create a new file path
    pdf_filepath = f"uploaded_pdf.pdf" 

    
    content = await file.read()

    with open("temp.pdf", "wb") as f:
        f.write(content)
    pdf_text = extract_text_from_pdf("temp.pdf")
    return {"pdf_text": pdf_text}

@app.post("/answer_questions/")
async def answer_questions(pdf_text: str, questions: list):
    cleaned_text = clean_text(pdf_text)
    chunks = split_into_chunks(cleaned_text)
    answers = []
    for question in questions:
        answer = ""
        for chunk in chunks:
            answer += answer_question(question, chunk) + "\n"
        answers.append({"question": question, "answer": answer.strip()})
    return {"answers": answers}
