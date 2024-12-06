import os
from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from a .env file
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Define Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question."),
        ("user", "Question: {question}")
    ]
)

# Streamlit Framework
st.title("Langchain Demo with Gemma Model")

# User input for question
input_text = st.text_input("What question do you have in mind?")

# Initialize Ollama's Gemma Model
try:
    llm = Ollama(model="gemma:2b")  # Properly initialize the Ollama instance
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    if input_text:
        # Process the input text
        st.write(chain.invoke({"question": input_text}))
except Exception as e:
    # Display error message if initialization fails
    st.error(f"Failed to initialize or process the model: {e}")