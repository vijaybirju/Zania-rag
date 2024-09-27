import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    with fitz.open(pdf_path) as pdf_doc:
        text = ""
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc.load_page(page_num)
            text += page.get_text("text")
    return text
