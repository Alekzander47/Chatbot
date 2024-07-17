import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os
import numpy as np

# Initialize the Streamlit app with session state to maintain chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Define a prompt template for the chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions."),
        ("user", "Question: {question}\n\nContext: {context}")
    ]
)

# Initialize the Ollama model
llm = Ollama(model="llama3")

# Create a chain that combines the prompt and the Ollama model
chain = prompt | llm

# Set up the Streamlit framework
st.title('Ask any questions if related to document')  # Set the title of the Streamlit app

# File uploader for documents
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

# Initialize the sentence transformer model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def get_document_embeddings(documents):
    embeddings = embedding_model.encode([document.text for document in documents])
    return np.array(embeddings)

def find_similar_documents(query, documents, doc_embeddings):
    query_embedding = embedding_model.encode([query]).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    ranked_indices = similarities.argsort()[::-1]
    return [documents[i] for i in ranked_indices[:5]]  # Retrieve top 5 documents

def perform_rag_query(documents, query):
    # Get document embeddings
    doc_embeddings = get_document_embeddings(documents)
    
    # Find similar documents
    similar_docs = find_similar_documents(query, documents, doc_embeddings)
    
    # Combine the content of similar documents into a single context
    context = "\n\n".join([doc.text for doc in similar_docs])
    return context

documents = []
response = ""

input_text = st.text_input("Ask your question!")  # Create a text input field in the Streamlit app

if uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to the temp directory
        for uploaded_file in uploaded_files:
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Load documents from the temp directory
        documents = SimpleDirectoryReader(temp_dir).load_data()

if input_text:
    try:
        context = ""
        # Perform RAG query if documents are uploaded
        if documents:
            context = perform_rag_query(documents, input_text)
        
        # Get model response
        model_response = chain.invoke({"question": input_text, "context": context})
        response = model_response

        # Append the user input and model response to the chat history
        st.session_state['chat_history'].append({"role": "user", "content": input_text})
        st.session_state['chat_history'].append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error invoking the model: {e}")
        st.error("Ensure the LLAMA3 model is properly pulled. Run 'ollama pull llama3' in the terminal.")
        response = ""

# Display the chat history in the main UI
for message in st.session_state['chat_history']:
    if message['role'] == "user":
        st.write(f"**User:** {message['content']}")
    else:
        st.write(f"**Assistant:** {message['content']}")

# Provide an additional prompt for the user to continue the conversation
if response:
    st.write("---")
    st.write("Ask another question and optionally upload documents for a document-based answer:")
    additional_input = st.text_input("Your question:", key="additional_input")
    additional_uploaded_files = st.file_uploader("Upload documents for this question", accept_multiple_files=True, key="additional_files")
    
    if additional_input:
        additional_documents = []
        if additional_uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to the temp directory
                for uploaded_file in additional_uploaded_files:
                    with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load documents from the temp directory
                additional_documents = SimpleDirectoryReader(temp_dir).load_data()
                
                # Perform RAG query with the additional documents
                context = perform_rag_query(additional_documents, additional_input)
        else:
            context = ""
        
        # Get model response
        model_response = chain.invoke({"question": additional_input, "context": context})
        response = model_response

        # Append the user input and model response to the chat history
        st.session_state['chat_history'].append({"role": "user", "content": additional_input})
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        
        st.experimental_rerun()

# Display the chat history in the sidebar
st.sidebar.title("Chat History")
for message in st.session_state['chat_history']:
    if message['role'] == "user":
        st.sidebar.write(f"**User:** {message['content']}")
    else:
        st.sidebar.write(f"**Assistant:** {message['content']}")
