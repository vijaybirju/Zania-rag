import fitz  

# Asynchronous function to extract text from a PDF
async def extract_text_from_pdf(pdf_path: str) -> str:
    pdf_text = ""
    pdf_document = fitz.open(pdf_path)
    
    # Extract text from each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pdf_text += page.get_text("text")
    
    pdf_document.close()
    return pdf_text
