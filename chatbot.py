from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# streamlit page configuration
st.set_page_config(
    page_title="Chatbot", 
    page_icon="🤖", 
    layout="centered"
)

st.title('Chatbot with Groq API')

# initiate chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# llm initialization
llm = ChatGroq(
    model="llama-3.3-70b-vesatile",
    temperature=0.0,
)

# handle user input and generate response
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    response = llm.invoke(
        input= [{"role": "user", "content": "You are a helpful assistant"}, *st.session_state.chat_history]
    )
    assistant_response = resonse.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

