import bs4
import  os
from langchain import hub
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv(override=True)
def create_vectorstore(chunks):
    """Create a vector store for efficient retrieval"""
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    )
    db = Chroma.from_documents(
        documents = chunks,
        embedding=embeddings,
        persist_directory="chroma_db_llamaparse1"
    )
    return db


def answer_question(question, text_chunks):
    """Answer a question using LangChain's QA chain"""
    vectorstore = create_vectorstore(text_chunks)
    llm = OpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    response = chain({"query": question})
    result = response['result']
    return result
