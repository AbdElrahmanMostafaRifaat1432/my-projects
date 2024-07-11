import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'

os.environ["GOOGLE_API_KEY"] = "AIzaSyDKlzT-aq7G6kT-ilENaX_HMfcZ149bnZw"

import os
import requests
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import torch
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain.utils.math import cosine_similarity
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
import streamlit as st
import time


def get_splits(url1):

    loader = WebBaseLoader(
        web_paths=(url1,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=True#("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    documents = splits  # Replace with your document list

    # Vectorstore creation with Chroma

    embeddings = GoogleGenerativeAIEmbeddings( model="models/embedding-001")

    vectorstore = Chroma.from_documents(documents=documents, embedding= embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4},)

    return retriever

def get_questions(llm , question):

    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
                        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
                        Generate multiple search queries related to: {question} \n
                        Output (2 queries): return only the questions """
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    
    # Chain
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    # Run
    questions = generate_queries_decomposition.invoke({"question":question})

    return questions




def rag_decomposition_with_routing(input):


    physics_template = """Here is the question you need to answer remember that You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner.:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    math_template = """Here is the question you need to answer remember that You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """


    # Embed prompts
    embeddings = GoogleGenerativeAIEmbeddings( model="models/embedding-001")
    prompt_templates = [physics_template, math_template]
    prompt_embeddings = embeddings.embed_documents(prompt_templates)



    # Embed question
    query_embedding = embeddings.embed_query(input["q_a_pairs"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt 
    st.write("Using physics prompt for decomposed question "+str(counter)+input['question'] if most_similar == physics_template else "Using math prompt for decomposed question "+str(counter)+input['question'])
    decomposition_prompt = PromptTemplate.from_template(most_similar)
    return decomposition_prompt



def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


st.title("URL Content Fetcher")

# Input text box for the URL
url = st.text_input("Enter the URL:").strip()

button1 = st.button("Fetch Content")

if button1 :
    if url:
        with st.spinner('Fetching content...'):

            if ('retriever' not in st.session_state) or (button1):
                st.session_state.retriever = get_splits(url)
                #st.session_state.question = st.text_input("Enter the question:")
                #st.session_state.button2 = st.button("answer")
            
            #retriever = get_splits(url)


api_key = "AIzaSyDKlzT-aq7G6kT-ilENaX_HMfcZ149bnZw"
llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-001", google_api_key=api_key , temperature=0.7)
        
        
question = st.text_input("Enter the question:")
button2 = st.button("answer")

if button2: #st.session_state.button2
    if question : #st.session_state.question
    
        questions = get_questions(llm , question) #st.session_state.question

        questions = [q for q in questions if q]

        #decomposition_prompt = rag_decomposition_with_routing()

        q_a_pairs = ""
        counter = 1
        for q in questions:
            
            rag_chain = (
            {"context": itemgetter("question") | st.session_state.retriever, 
            "question": itemgetter("question"),
            "q_a_pairs": itemgetter("q_a_pairs")} 
            | RunnableLambda(rag_decomposition_with_routing)
            | llm
            | StrOutputParser())

            answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
            q_a_pair = format_qa_pair(q,answer)
            q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

            #time.sleep(4)
            counter+=1
        st.write(answer)

    else:
        st.warning("Please enter a valid URL.")

    

