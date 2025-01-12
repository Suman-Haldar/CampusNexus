import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import os

API_KEY = st.secrets["API_KEY"]
TXT_FILE = "db.txt"  # Text file containing knowledge base content

# Validate API key
if not API_KEY or not API_KEY.startswith("sk-"):
    st.error("OpenAI API Key is missing or invalid. Please set the API_KEY in Streamlit secrets.")
    st.stop()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = API_KEY

# Check if the TXT file exists
if not os.path.exists(TXT_FILE):
    st.error(f"Text file '{TXT_FILE}' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the BOT_PROMPT as a template for response generation
BOT_PROMPT = """
You are Campus Nexus, an intelligent assistant designed to help students, faculty, and visitors of Priyadarshini Engineering College. Your primary goal is to provide accurate, informative, and helpful responses to queries related to the college, its programs, facilities, and campus life, while maintaining a friendly and professional tone.
Instructions:

Greet the user warmly and introduce yourself as Campus Nexus, the intelligent assistant for Priyadarshini Engineering College.
Ask the user how you can assist them with any questions or information they need about the college.
Provide clear, concise, and easy-to-understand explanations about the college's academic programs, admission processes, campus facilities, events, faculty, or any other relevant topics.
If the user's query is complex or multi-faceted, break down your response into logical steps or points.
If the user's query falls outside of your knowledge domain about Priyadarshini Engineering College, politely inform them that you don't have that specific information.
If the provided context does not contain relevant information to answer the user's query, politely respond with "I'm afraid I don't have enough information to answer that question about Priyadarshini Engineering College."
Maintain a friendly, helpful, and professional tone throughout the conversation, reflecting the values and spirit of Priyadarshini Engineering College.
Offer to provide additional information or clarification if the user needs it.

Anything between the <context> html blocks is the only information you should rely on to answer questions about Priyadarshini Engineering College.
<context>
{context}
</context>
REMEMBER: Respond only in English, without displaying any HTML tags or retrieved knowledge blocks. Present the information naturally as if it is your own integrated knowledge about Priyadarshini Engineering College. Do not reference the context or outside sources in your responses.
"""

# Load Text and Create FAISS Knowledge Base
@st.cache_resource
def create_knowledge_base():
    try:
        # Read the TXT file
        with open(TXT_FILE, "r", encoding="utf-8") as file:
            text = file.read()

        # Split text for better embedding performance
        text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=500)  # Adjusted chunk size and overlap
        split_docs = text_splitter.split_text(text)

        # Convert text chunks to Document objects
        documents = [Document(page_content=chunk) for chunk in split_docs]

        # Create embeddings and FAISS vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating knowledge base: {str(e)}")
        st.stop()

# Initialize Chatbot Chain
@st.cache_resource
def initialize_chatbot(_vector_store):
    try:
        llm = ChatOpenAI(temperature=0)  # Use OpenAI GPT model
        chain = ConversationalRetrievalChain.from_llm(llm, retriever=_vector_store.as_retriever())
        return chain
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.stop()

# Set up Streamlit App
st.set_page_config(page_title="Campus Nexus Chatbot", layout="wide")
st.title("Campus Nexus - Chatbot for Priyadarshini Engineering College")
st.markdown("""
Welcome to **Campus Nexus**, your intelligent assistant for Priyadarshini Engineering College.  
Ask questions about the college, and I'll provide context-aware responses.
""")

# Add Sidebar with information about the bot
with st.sidebar:
    st.header("About Campus Nexus")
    st.write("""
    **Campus Nexus** is an intelligent chatbot assistant designed to help students, faculty, and visitors of Priyadarshini Engineering College.  
    It provides accurate and informative responses regarding:
    - College programs
    - Admission processes
    - Campus facilities
    - Events
    - Faculty
    - And much more related to college life.

    Ask me anything, and I'll do my best to assist you!
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load FAISS Knowledge Base and Initialize Chatbot
vector_store = create_knowledge_base()
chatbot_chain = initialize_chatbot(vector_store)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything about Priyadarshini Engineering College..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Get chatbot response
        with st.chat_message("assistant"):
            # Modify the prompt to include context and instructions
            modified_prompt = BOT_PROMPT.format(context=prompt)
            response = chatbot_chain({"question": prompt, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Update chat history for retrieval
            st.session_state.chat_history.append((prompt, answer))
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
