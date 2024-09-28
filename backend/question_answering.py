import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import  os
from groq import Groq
# from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import OpenAIEmbeddings
from llama_index.core import Settings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv(override=True)
print("load_dotenv",load_dotenv())
api_key = os.getenv("GROQ_API_KEY")

def create_vectorstore(chunks):
    """Create a vector store for efficient retrieval"""
    # # embeddings = OpenAIEmbeddings()
    # if os.path.exists(CHROMA_PATH):
    #     shutil.rmtree(CHROMA_PATH)
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
    )
    # Create a new Chroma database from the documents using OpenAI embeddings
    # embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma.from_documents(
        documents = chunks,
        embedding=embeddings,
        persist_directory="chroma_db_llamaparse1"
    )
    # collection = create_or_update_collection(db, embeddings)
    # Persist the database to disk
    # db.persist()
    return db
# def create_or_update_collection(db, embeddings):
#     """Create or update the collection in Chroma."""
#     client = db.PersistentClient(path='db')
#     # Specify a unique name for the collection
#     collection_name = "langchain"  # Update this if needed
#     # Get or create the collection
#     collection = client.get_collection(name=collection_name, embedding_function=embeddings)

#     return collection


def answer_question(question, text_chunks):
    """Answer a question using LangChain's QA chain"""
    vectorstore = create_vectorstore(text_chunks)
    # llm = OpenAI(model_name="text-davinci-003")
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    response = chain({"query": question})
    result = response['result']
    return result
