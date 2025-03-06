import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
import google.generativeai as genai
import base64
import streamlit.components.v1 as components  

def load_gemini_api_key():
    dotenv_path = "gemini.env"  
    load_dotenv(dotenv_path)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError(f"Unable to retrieve GEMINI_API_KEY from {dotenv_path}")
    return gemini_api_key


def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase


def summarize_with_gemini(text):
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"Summarize the following text in 3-5 sentences:\n\n{text}"
    
    response = model.generate_content(prompt)
    return response.text




def main():
    
    
    background_image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQiRFApPcHG-vyXaXhj4MB8yhREhXSmWOsjkA&s"

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
       
        <style>
            body {{
                font-family: Arial, sans-serif;
                
                padding: 0;
                background-image: url('{background_image_url}');
                
                background-repeat: no-repeat;
                background-position: center;
                height: 200px;
                width: 100vw;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
        
         
        </style>
    </head>
    
    </html>
    """
    
    # Render the HTML template
    components.html(html_template, height=200)
    st.title("📄 AP Summarizer")
    st.write("Powered by CTI")
    st.divider()
    
    

    try:
        os.environ["GEMINI_API_KEY"] = load_gemini_api_key()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except ValueError as e:
        st.error(str(e))
        return

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

        knowledgeBase = process_text(text)

        query = "Summarize the content of the uploaded PDF file in approximately 3-5 sentences. Focus on capturing the main ideas and key points discussed in the document."

        if query:
            docs = knowledgeBase.similarity_search(query)
            combined_text = " ".join([doc.page_content for doc in docs])

            # Use Gemini instead of OpenAI
            response = summarize_with_gemini(combined_text)

            st.subheader('Summary Results:')
            st.write(response)

if __name__ == '__main__':
    main()
