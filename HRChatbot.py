import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import streamlit as st
from IPython.display import display
from IPython.display import Markdown
import textwrap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from llm_guard.input_scanners import Toxicity
from llm_guard.input_scanners.toxicity import MatchType
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType

st.header("HR ChatBot")
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')
llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader("ZETA_CORPORATION.pdf")

    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    return documents
pdf_path = "ZETA_CORPORATION.pdf"
documents = load_and_split_pdf(pdf_path)
@st.cache_resource

def vector():
    llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever(search_kwargs={"k": 10})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever,
    search_kwargs={"k": 8})
    llm = llm
    compressor = CohereRerank(top_n=5)
    rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=compression_retriever
    )
    return rerank_retriever
rerank_retriever=vector()
output_parser = StrOutputParser()
llm=ChatGoogleGenerativeAI(model='gemini-pro',convert_system_message_to_human=True,temperature=0.5)
instruction_to_system = """
Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is.
"""

question_maker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_to_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


question_chain = question_maker_prompt | llm | StrOutputParser()
# Use three sentences maximum and keep the answer concise.\
qa_system_prompt = """be act like a HR officer an answer the questions to the employye
{context}
Question: {question}
Helpful Answer:"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
def contextualized_question(input: dict):
    if input.get("chat_history"):
        return question_chain
    else:
        return input["question"]
retriever_chain = RunnablePassthrough.assign(
        context=contextualized_question | rerank_retriever #| format_docs
    )
rag_chain = (
    retriever_chain
    | qa_prompt
    | llm
)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to reset chat history
def reset_chat_history():
    st.session_state.chat_history = []
# New Chat button
if st.button('New Chat'):
    reset_chat_history()
# Custom CSS to style the messages
st.markdown("""
    <style>
    .bot-message {
        text-align: left;
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        color: black;
    }
    .human-message {
        text-align: right;
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        color: black;
    }
    
    </style>
    </style>
    """, unsafe_allow_html=True)

# Display chat history
for i in range(0, len(st.session_state.chat_history), 2):
    human_message = st.session_state.chat_history[i].content
    ai_message = st.session_state.chat_history[i+1].content if i+1 < len(st.session_state.chat_history) else ''
    st.markdown(f"<div class='human-message'><strong>You:</strong> {human_message}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-message'><strong>Bot:</strong> {ai_message}</div>", unsafe_allow_html=True)

query = st.text_input('Enter the query')
question = ''

@st.cache_resource
def scanner():
    scanner=Toxicity(threshold=0.5, match_type=MatchType.SENTENCE)
    return scanner
scanner=scanner()
@st.cache_resource
def scanner1():
    scanner1 = PromptInjection(threshold=0.5, match_type=MatchType.FULL)
    return scanner1
scanner1=scanner1()
def answer_question(question):
    recent_history = st.session_state.chat_history[-14:] if len(st.session_state.chat_history) > 14 else st.session_state.chat_history
    ai_msg = rag_chain.invoke({"question": question, "chat_history": recent_history})
    st.session_state.chat_history.extend([HumanMessage(content=question), ai_msg])
    if len(st.session_state.chat_history) > 14:
        st.session_state.chat_history = st.session_state.chat_history[-14:]
    return ai_msg.content
    

if st.button('➤'):
    sanitized_prompt, is_valid, risk_score = scanner.scan(query)

    if is_valid and risk_score < 0.5:  # Adjust the threshold as needed
            sanitized_prompt, is_valid, risk_score = scanner1.scan(query)
            if is_valid and risk_score < 0.5:  # Adjust the threshold as needed
                        question=query
            else:
                st.write("queetion is towards prompt injection")   
    else:
        st.write("question is either invalid or toxic.")
    if question:
        result = answer_question(question)
        st.markdown(result)
    

