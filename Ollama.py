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
st.set_page_config(page_title="Langchain + Ollama Demo", layout="centered")
st.title("Langchain Demo with Gemma Model")
st.markdown("### A Streamlit app demonstrating Langchain's integration with Ollama's Gemma model.")

# Input Section
with st.sidebar:
    st.header("Model Information")
    st.markdown(
        """
        - **Model**: Gemma 2B
        - **Provider**: Ollama
        - **Purpose**: General-purpose LLM for question answering.
        """
    )

st.write("### Ask a Question")
input_text = st.text_input("Enter your question below:", key="user_input")
submit_button = st.button("Get Response")

# Display Status
if submit_button:
    if input_text:
        try:
            # Initialize Ollama's Gemma Model
            st.write("Processing your question...")
            llm = Ollama(model="gemma:2b")  # Properly initialize the Ollama instance
            output_parser = StrOutputParser()
            chain = prompt | llm | output_parser

            # Get and display the response
            response = chain.invoke({"question": input_text})
            st.success("Response generated successfully!")
            st.write("### Response:")
            st.write(response)
        except Exception as e:
            # Display error message if initialization fails
            st.error(f"Failed to initialize or process the model: {e}")
    else:
        st.warning("Please enter a question before submitting.")
else:
    st.info("Waiting for input...")

# Footer Section
st.markdown(
    """
    ---
    **Powered by:** [Langchain](https://www.langchain.com/) | [Streamlit](https://streamlit.io/)
    """
)
