import bs4
import  os
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import faiss
import numpy as np


load_dotenv(override=True)

client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def create_vectorstore(chunks):
    """Create a Faiss vector store for efficient retrieval"""
    # Initialize Faiss index with the appropriate dimension (based on the embedding size)
    dimension = len(get_embedding(chunks[0]))  
    index = faiss.IndexFlatL2(dimension)
    
    # Store all embeddings and corresponding text chunks
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        index.add(np.array(embedding).reshape(1, -1))
    return index, chunks  # Return the index and the text chunks for retrieval


async def initialize_openai_client():
    """Initialize the OpenAI client"""
    return AsyncOpenAI()

def initialize_vectorstore(text_chunks):
    """Create and return the vector store for the given text chunks"""
    vectorstore, original_chunks = create_vectorstore(text_chunks)
    return vectorstore, original_chunks

def get_relevant_chunk(question, vectorstore, original_chunks):
    """Get the relevant chunk from the text using vector search."""
    question_embedding = np.array(get_embedding(question), dtype='float32').reshape(1, -1)
    distances, indices = vectorstore.search(question_embedding, k=3)
    relevant_chunks = [original_chunks[idx] for idx in indices[0]]
    combined_relevant_chunk = "\n".join(relevant_chunks)  # You can choose to join with a different separator if needed

    return combined_relevant_chunk

async def fetch_answer_from_openai(question, relevant_chunk, client):
    """Generate the answer using OpenAI's API"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {relevant_chunk}\n\nQuestion: {question}\nAnswer as concisely as possible."}
    ]
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=messages,
        max_tokens=150,
        n=1,
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

async def answer_question(question, vectorstore, original_chunks, client):
    """Answer a question using vector search and OpenAI API"""
    relevant_chunk = get_relevant_chunk(question, vectorstore, original_chunks)
    answer = await fetch_answer_from_openai(question, relevant_chunk, client)

    return answer
