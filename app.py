import streamlit as st
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
import pinecone
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Disable certain PyTorch features that cause issues with Streamlit on Python 3.13.2
import sys
if sys.version_info >= (3, 13):
    os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'
    # This helps avoid PyTorch path issues
    os.environ['PYTORCH_JIT'] = '0'

# Load environment variables
load_dotenv()

# Initialize clients
@st.cache_resource
def initialize_resources():
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = pc.Index("web-content-index")
    return pc, model, index

pc, model, index = initialize_resources()

# Define the state type
class AgentState(TypedDict):
    query: str
    documents: List[str]
    response: Optional[str]
    is_related: Optional[bool]

# Node 1: Check if the query is related to Occam's Advisory
def check_query(state: AgentState) -> dict:
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a decision-making assistant. Determine if the user's query pertains to the Occam's Advisory website content. Respond with 'yes' or 'no'."),
        ("human", "{query}")
    ])
    chain = prompt | llm
    ai_msg = chain.invoke({"query": state["query"]})
    is_related = ai_msg.content.strip().lower() == 'yes'
    return {"is_related": is_related}

# Node 2: Retrieve relevant documents based on the query
def retrieve(state: AgentState) -> dict:
    query_embedding = model.encode(state["query"]).tolist()
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    documents = [m.metadata.get("text") for m in results.matches]
    return {"documents": documents}

# Node 3: Evaluate the relevance of retrieved documents
def evaluate(state: AgentState) -> dict:
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Assess if the retrieved document is relevant to the user's query. Respond with 'yes' or 'no'."),
        ("human", "User Query: {query}\nRetrieved Document: {document}")
    ])
    chain = prompt | llm
    relevant_docs = []
    for doc in state["documents"]:
        ai_msg = chain.invoke({"query": state["query"], "document": doc})
        if ai_msg.content.strip().lower() == 'yes':
            relevant_docs.append(doc)
    return {"documents": relevant_docs}

# Node 4: Refine the user's query if no relevant documents are found
def refine(state: AgentState) -> dict:
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "The current query did not yield relevant results. Suggest a refined query."),
        ("human", "Original query: {query}")
    ])
    chain = prompt | llm
    ai_msg = chain.invoke({"query": state["query"]})
    refined_query = ai_msg.content.strip()
    return {"query": refined_query}

# Node 5: Generate a response based on relevant documents
def respond(state: AgentState) -> dict:
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    context = " ".join(state["documents"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Using the provided context, respond to the user's query."),
        ("human", "Context: {context}\nQuery: {query}")
    ])
    chain = prompt | llm
    ai_msg = chain.invoke({"context": context, "query": state["query"]})
    response = ai_msg.content.strip()
    return {"response": response}

# Node 6: Handle unrelated queries using a general-purpose LLM
def general_llm_response(state: AgentState) -> dict:
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{query}")
    ])
    chain = prompt | llm
    ai_msg = chain.invoke({"query": state["query"]})
    response = ai_msg.content.strip()
    return {"response": response}

# Initialize the workflow
@st.cache_resource
def create_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes to the workflow
    workflow.add_node('check_query', check_query)
    workflow.add_node('retrieve', retrieve)
    workflow.add_node('evaluate', evaluate)
    workflow.add_node('refine', refine)
    workflow.add_node('respond', respond)
    workflow.add_node('general_llm_response', general_llm_response)
    
    # Define edges between nodes
    workflow.set_entry_point('check_query')
    workflow.add_conditional_edges(
        'check_query',
        lambda state: 'retrieve' if state['is_related'] else 'general_llm_response'
    )
    workflow.add_edge('retrieve', 'evaluate')
    workflow.add_conditional_edges(
        'evaluate',
        lambda state: 'respond' if state['documents'] else 'refine'
    )
    workflow.add_edge('refine', 'retrieve')
    workflow.add_edge('respond', END)
    workflow.add_edge('general_llm_response', END)
    
    # Compile the workflow
    return workflow.compile()

def process_query(query):
    try:
        app = create_workflow()
        initial_state = AgentState(query=query, documents=[], response=None, is_related=None)
        final_state = app.invoke(initial_state)
        return final_state
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return AgentState(
            query=query,
            documents=[],
            response=f"Sorry, I encountered an error while processing your question. Please try again or rephrase your question. Error: {str(e)}",
            is_related=False
        )

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Occam's Advisory Q&A",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Occam's Advisory Q&A Assistant")
    st.markdown("""
    Ask a question about Occam's Advisory or any other topic. The assistant will retrieve relevant information 
    or provide a general response if the question is not related to Occam's Advisory.
    """)
    
    # Session state to track question history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # User input
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_input("Type your question:", key="user_input")
        submit_button = st.form_submit_button("Ask")
        
    # Process the question when submitted
    if submit_button and user_question:
        try:
            with st.spinner("Thinking..."):
                # Add a progress indicator for each step
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Process stages with progress updates
                progress_text.text("Checking if question is related to Occam's Advisory...")
                progress_bar.progress(20)
                
                final_state = process_query(user_question)
                
                progress_bar.progress(80)
                progress_text.text("Retrieving and processing information...")
                
                # Add the Q&A pair to history
                st.session_state.history.append({
                    "question": user_question,
                    "answer": final_state["response"],
                    "is_related": final_state.get("is_related", False),
                    "docs_used": len(final_state.get("documents", []))
                })
                
                progress_bar.progress(100)
                progress_text.empty()
                progress_bar.empty()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Display history in reverse order (newest first)
    if st.session_state.history:
        st.markdown("## Question History")
        
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q: {item['question']}", expanded=(i == 0)):
                st.markdown(f"**Answer:**\n{item['answer']}")
                
                # Show metadata about the response
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Related to Occam's Advisory: {'Yes' if item['is_related'] else 'No'}")
                with col2:
                    if item['is_related']:
                        st.info(f"Documents used: {item['docs_used']}")
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This Q&A assistant uses a LangGraph workflow to:
        
        1. Check if your query is related to Occam's Advisory
        2. Retrieve and evaluate relevant documents
        3. Generate a response based on the retrieved information
        4. Refine queries if no relevant information is found
        
        For non-Occam's Advisory questions, it uses a general LLM.
        """)
        
        st.info(f"Running on Python {sys.version}")

if __name__ == "__main__":
    main()