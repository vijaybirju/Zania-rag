import re

def clean_text(text: str) -> str:
    # Remove unwanted characters, normalize whitespace, etc.
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_chunks(text: str, chunk_size=500):
    # Split text into smaller chunks for better question answering
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
