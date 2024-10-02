import re
from langchain.text_splitter import CharacterTextSplitter


def split_into_chunks(text: str, chunk_size=500, overlap=200):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        # Create a chunk of text
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
