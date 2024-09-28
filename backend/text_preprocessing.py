import re
from langchain.text_splitter import CharacterTextSplitter

# def clean_text(docs: list) -> list:
#     # Remove unwanted characters, normalize whitespace, etc.
#     docs_ = []
#     for doc in docs:
#         doc = re.sub(r'\s+', ' ', doc)
#         docs_.append(doc)
#     return docs_

def split_into_chunks(docs, chunk_size=500):
    # Split text into smaller chunks for better question answering
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
        

    # # <<<<<<<<<<<<<<<<<<<<<<<<<<  To use ChromaDB >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    docs = text_splitter.split_documents(docs)
    return docs
