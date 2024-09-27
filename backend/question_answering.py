from transformers import pipeline

qa_model = pipeline("question-answering")

def answer_question(question: str, context: str) -> str:
    result = qa_model(question=question, context=context)
    return result['answer']
