# PDF Question Answering System

This project allows users to upload a PDF document and ask questions related to the content of the PDF. The system will extract text from the PDF, process it, and provide answers to the user's questions using a Language Model (LLM).

## Features

- Upload PDF files for content extraction
- Input multiple questions (one per line)
- Retrieves answers for each question from the content of the uploaded PDF
- Uses OpenAI's `text-davinci-003` model for generating responses
- Simple and interactive web interface built using [Streamlit](https://streamlit.io/)

## Prerequisites

To run this project, ensure that you have the following installed:

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/installation/)
- Streamlit
- OpenAI API key (for using the `text-davinci-003` model)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/pdf-question-answering-system.git
   cd pdf-question-answering-system
