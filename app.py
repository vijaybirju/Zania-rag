from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# from .fastapi run main.pybackend.pdf_processing import extract_text_from_pdf
from backend import question_answering, text_preprocessing
from backend.question_answering import answer_question
from backend.text_preprocessing import  clean_text, split_into_chunks


app = FastAPI()

pdf_filepath = None 

# HTML template rendering using Jinja2
templates = Jinja2Templates(directory="frontend/templates")

@app.get("/", response_class=HTMLResponse)
async def render_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_filepath 
    
    # Check if a previous PDF exists and delete it
    if pdf_filepath:
        os.remove(pdf_filepath)
    
    # Create a new file path
    pdf_filepath = f"temp.pdf"
    content = await file.read()
    with open(pdf_filepath, "wb") as f:
        f.write(content)
    pdf_text = extract_text_from_pdf(pdf_filepath)
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

@app.post("/process_form/", response_class=HTMLResponse)
async def process_form(request: Request, file: UploadFile = File(...), questions: str = ""):
    # Process uploaded file and answer questions
    content = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(content)
    
    # Extract and clean PDF text
    pdf_text = extract_text_from_pdf("temp.pdf")
    cleaned_text = clean_text(pdf_text)
    chunks = split_into_chunks(cleaned_text)
    
    # Split questions by line
    question_list = questions.splitlines()
    
    # Answer questions
    answers = []
    for question in question_list:
        answer = ""
        for chunk in chunks:
            answer += answer_question(question, chunk) + "\n"
        answers.append({"question": question, "answer": answer.strip()})
    
    return templates.TemplateResponse("index.html", {"request": request, "answers": answers})
