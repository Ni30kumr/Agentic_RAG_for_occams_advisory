from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
import pinecone
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Initialize clientspy
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = pc.Index("web-content-index")

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
app = workflow.compile()

# Example usage
initial_state = AgentState(query="Services offered by Occams Advisory?", documents=[], response=None, is_related=None)

# Invoke the workflow
final_state = app.invoke(initial_state)

# Print the final response
print("Final Response:", final_state["response"])
