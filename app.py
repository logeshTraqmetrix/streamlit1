# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# import gradio as gr

# # Initialize embeddings and load vector store
# embeddings = OllamaEmbeddings(model='mxbai-embed-large')
# db = FAISS.load_local('merged1_vectors', embeddings, allow_dangerous_deserialization=True)

# # Initialize Groq chat model - using a more capable model
# groq_chat = ChatGroq(
#     temperature=0.7, 
#     model_name="llama-3.3-70b-versatile",
#     api_key=
# )

# # Create a retriever with better search parameters
# retriever = db.as_retriever(
#     search_type="mmr",  # Use Max Marginal Relevance for better diversity
#     search_kwargs={
#         "k": 8,  # Increase number of documents retrieved
#         "fetch_k": 20,  # Larger pool for MMR to select from
#         "lambda_mult": 0.5  # Balance between relevance and diversity
#     }
# )

# # Enhanced prompt template
# prompt_template = """You are an expert assistant trained to provide detailed, helpful answers based on the provided context.

# Context information is below:
# ---------------------
# {context}
# ---------------------

# Given the context information and no prior knowledge, answer the question: {question}

# Your response should:
# - Be comprehensive and detailed
# - Include necessary relevant information from the context
# - Use bullet points or numbered lists when appropriate
# - If steps are involved, explain each step clearly

# If the context doesn't contain the answer, say "I couldn't find this information in the documents.
# """
# PROMPT = PromptTemplate(
#     template=prompt_template, 
#     input_variables=["context", "question"]
# )

# # Create retrieval QA chain with better configuration
# qa_chain = RetrievalQA.from_chain_type(
#     llm=groq_chat,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={
#         "prompt": PROMPT,
#         "verbose": True  # For debugging
#     },
#     verbose=True
# )

# def respond(message, history):
#     # Get response from QA chain
#     result = qa_chain({"query": message})
#     answer = result["result"]
    
#     # Process source documents more thoroughly
#     sources = []
#     for doc in result['source_documents']:
#         source = doc.metadata.get('source', 'Unknown source')
#         content = doc.page_content[:150] + "..."  # Show snippet of content
#         sources.append(f"{source}: {content}")
    
#     return answer

# # Enhanced Gradio interface
# demo = gr.ChatInterface(
#     respond,
#     chatbot=gr.Chatbot(height=520, render_markdown=True),  # Enable markdown rendering
#     textbox=gr.Textbox(placeholder="Ask me anything about your documents...", 
#                       container=False, 
#                       scale=7,
#                       autofocus=True),
#     title="Advanced Document QA Assistant",
#     description="Ask detailed questions about your documents. The AI will provide comprehensive answers drawn from your knowledge base.",
#     theme="soft",
#     examples=[
#         ["Explain how to create a new form workflow in detail"]
#     ],
#     cache_examples=False,
# )

# if __name__ == "__main__":
#     demo.launch()
















# app.py
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Set page config
st.set_page_config(
    page_title="Zoho Creator Assistant",
    page_icon="📚",
)

# Initialize with Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize embeddings and load vector store
@st.cache_resource
def load_vectordb():
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    db = FAISS.load_local('merged1_vectors', embeddings, allow_dangerous_deserialization=True)
    return db

try:
    db = load_vectordb()
except Exception as e:
    st.error(f"Failed to load vector database: {e}")
    st.stop()

# Get API key from environment variables
# groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key = st.secrets["GROQ_API_KEY"]
if not groq_api_key:
    st.error("GROQ_API_KEY environment variable not set")
    st.stop()

# Initialize Groq chat model
groq_chat = ChatGroq(
    temperature=0.7, 
    model_name="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

# Create retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

# Enhanced prompt template
prompt_template = """You are an expert assistant trained to provide detailed, helpful answers based on the provided context.

Context information is below:
---------------------
{context}
---------------------

Given the context information and no prior knowledge, answer the question: {question}

Your response should:
- Be comprehensive and detailed
- Include necessary relevant information from the context
- Use bullet points or numbered lists when appropriate
- If steps are involved, explain each step clearly

If the context doesn't contain the answer, say "I couldn't find this information in the documents."
"""
PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# Create retrieval QA chain
@st.cache_resource
def get_qa_chain():
    return RetrievalQA.from_chain_type(
        llm=groq_chat,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": True
        },
        verbose=True
    )

qa_chain = get_qa_chain()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain({"query": prompt})
                answer = result["result"]
                
                # Display assistant response
                st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
