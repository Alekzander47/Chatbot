from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st

# Define a prompt template for the chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions."),
        ("user", "Question:{question}")
    ]
)

# Set up the Streamlit framework
st.title('Ask any questions <-_->')  # Set the title of the Streamlit app
input_text = st.text_input("Ask your question!")  # Create a text input field in the Streamlit app

# Initialize the Ollama model
llm = Ollama(model="llama3")

# Create a chain that combines the prompt and the Ollama model
chain = prompt | llm

# Invoke the chain with the input text and display the output
if input_text:
    try:
        response = chain.invoke({"question": input_text})
        st.write(response)
    except Exception as e:
        st.error(f"Error invoking the model: {e}")
        st.error("Ensure the LLAMA2 model is properly pulled. Run 'ollama pull llama2' in the terminal.")