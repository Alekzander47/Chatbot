from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from llama_index.core import SimpleDirectoryReader, Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np

# Define the prompt and model
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions."),
        ("user", "Question: {question}\n\nContext: {context}")
    ]
)
llm = Ollama(model="llama3")
chain = prompt | llm
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def get_document_embeddings(documents):
    embeddings = embedding_model.encode([document.text for document in documents])
    return np.array(embeddings)

def find_similar_documents(query, documents, doc_embeddings):
    query_embedding = embedding_model.encode([query]).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    ranked_indices = similarities.argsort()[::-1]
    return [documents[i] for i in ranked_indices[:5]]

def perform_rag_query(documents, query):
    if not documents:
        return ""
    doc_embeddings = get_document_embeddings(documents)
    similar_docs = find_similar_documents(query, documents, doc_embeddings)
    context = "\n\n".join([doc.text for doc in similar_docs])
    return context

from allauth.socialaccount.models import SocialAccount

@login_required
def index(request):
    profile_picture_url = None
    social_account = SocialAccount.objects.filter(user=request.user, provider='google').first()
    if social_account:
        profile_picture_url = social_account.extra_data.get('picture')

    if request.method == 'POST':
        try:
            documents = []
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_uploaded_files')
            if os.path.exists(temp_dir) and os.listdir(temp_dir):
                documents = SimpleDirectoryReader(temp_dir).load_data()

            data = json.loads(request.body)
            message = data.get('message', '')

            context = perform_rag_query(documents, message) if documents else ""

            response = chain.invoke({"question": message, "context": context})

            return JsonResponse({'response': response})
        except Exception as e:
            return JsonResponse({'response': f'An error occurred: {e}'}, status=500)
    
    context = {
        'profile_picture_url': profile_picture_url
    }
    return render(request, 'admin_view/index.html', context)

def upload_files(request):
    if request.method == 'POST':
        try:
            uploaded_files = request.FILES.getlist('uploaded_files')
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_uploaded_files')

            # Create the temporary directory if it doesn't exist
            os.makedirs(temp_dir, exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)
            
            return JsonResponse({'status': 'success', 'message': 'Files uploaded successfully'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'File upload failed: {e}'}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

def logout_view(request):
    logout(request)
    return redirect('/')

def signup(request):
    return render(request, 'account/signup.html')
