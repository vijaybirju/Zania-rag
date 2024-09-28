# import fitz  # PyMuPDFSS
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs