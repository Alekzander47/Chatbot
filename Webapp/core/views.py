from django.shortcuts import render
from django.http import JsonResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import tempfile

# Create your views here.
def index(request):
    return render(request, 'admin/index.html')

def login(request):
    return render(request, 'admin/login.html')

def signup(request):
    return render(request, 'admin/signup.html')


# Initialize the sentence transformer model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Define a prompt template for the chatbot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the questions."),
    ("user", "Question: {question}\n\nContext: {context}")
])

# Initialize the Ollama model
llm = Ollama(model="llama3")

# Create a chain that combines the prompt and the Ollama model
chain = prompt | llm

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


def ask_question(request):
    response_data = {'response': ''}
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        documents = []

        if request.FILES:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in request.FILES.getlist('documents'):
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        for chunk in uploaded_file.chunks():
                            f.write(chunk)
                
                documents = SimpleDirectoryReader(temp_dir).load_data()

        if input_text:
            try:
                context = ""
                if documents:
                    context = perform_rag_query(documents, input_text)

                model_response = chain.invoke({"question": input_text, "context": context})
                response_data['response'] = model_response

            except Exception as e:
                response_data['error'] = str(e)

    return JsonResponse(response_data)
