import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile

# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Set language model
Settings.llm = Ollama(model="llama3")

# Function to create document index from uploaded files
def create_index_from_files(uploaded_files):
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        documents = SimpleDirectoryReader(temp_dir).load_data()
        return VectorStoreIndex.from_documents(documents)

# Define a prompt template for the chatbot
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions."),
        ("user", "Question: {question}")
    ]
)

# Set up the Streamlit framework
st.title('Ask any questions')

# File uploader for documents
uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["txt", "md", "pdf", "docx", "html"])

# Text input for questions
input_text = st.text_input("Ask your question!")

# Initialize the Ollama model
llm = Ollama(model="llama3")

# Function to handle the response
def get_response(input_text, index):
    try:
        if index:
            response = index.as_query_engine().query(input_text)
            if response:
                return response
        # Fall back to regular LLM response if no document response or no index
        formatted_prompt = prompt_template.format(question=input_text)
        response = llm(formatted_prompt)
        return response
    except Exception as e:
        return f"Error invoking the model: {e}\nEnsure the LLAMA3 model is properly pulled. Run 'ollama pull llama3' in the terminal."

# Initialize the document index if files are uploaded
index = None
if uploaded_files:
    index = create_index_from_files(uploaded_files)

# Display the response
if input_text:
    response = get_response(input_text, index)
    st.write(response)
