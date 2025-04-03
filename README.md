# Agentic RAG for Occam's Advisory Q&A

This repository contains an implementation of an Agentic Retrieval Augmented Generation (RAG) system specifically designed for querying information related to Occam's Advisory. It includes scripts for scraping website content, processing it, creating a vector database, and deploying a Streamlit application for user interaction.

## File Significance

* **`Scraper.py`:**
    * This script is responsible for scraping content from the Occam's Advisory website using its sitemap.
    * It extracts relevant text, metadata (like titles and URLs), and cleans the content.
    * It saves the scraped data in a JSON format (`occams_content.json`).
    * It includes robust error handling and retry logic to ensure reliable data collection.
* **`clean_txt.py`:**
    * Processes the `occams_content.json` file.
    * Extracts the cleaned text from the JSON and saves it into `occams_clean_text.txt`
* **`setup_vector_database.py`:**
    * This file is used to setup the vector database.
    * It takes the cleaned text from `occams_clean_text.txt` and creates embeddings which are stored in Pinecone vector database.
* **`main.py`:**
    * Implements the core Agentic RAG workflow using LangGraph.
    * Defines the nodes and edges of the graph, including query checking, document retrieval, relevance evaluation, query refinement, and response generation.
    * Uses Pinecone for vector search and Groq's LLM for response generation.
    * This is the core logic that the app.py file utilizes.
* **`app.py`:**
    * Creates a Streamlit web application for user interaction.
    * Integrates the Agentic RAG workflow from `main.py`.
    * Provides a user-friendly interface for querying Occam's Advisory information.
    * Displays the query history and relevant metadata.
    * Provides a user friendly UI.
* **`requirements.txt`:**
    * Lists the Python packages required to run the application.
* **`.env`:**
    * Stores sensitive information like API keys (Pinecone, Groq). This file is not commited to the repository for security reasons.

## Agentic RAG Speciality

This implementation goes beyond a simple RAG system by incorporating agentic capabilities. Here's how it differs:

* **Query Understanding and Routing:**
    * The `check_query` node determines if a user's query is relevant to Occam's Advisory. This allows the system to route irrelevant queries to a general-purpose LLM, preventing unnecessary vector searches.
* **Relevance Evaluation:**
    * The `evaluate` node assesses the relevance of retrieved documents. This filters out irrelevant information, ensuring that the LLM only uses highly relevant context for response generation.
* **Query Refinement:**
    * The `refine` node suggests refined queries if no relevant documents are found. This helps improve retrieval accuracy and provides a better user experience.
* **Workflow Orchestration:**
    * LangGraph enables the creation of a dynamic workflow with conditional edges. This allows the system to adapt its behavior based on the query and retrieved documents.

**Full Flow:**

1.  **User Query:** The user enters a question in the Streamlit app.
2.  **Query Check:** The `check_query` node uses an LLM to determine if the query is related to Occam's Advisory.
3.  **Routing:**
    * If related, the workflow proceeds to document retrieval.
    * If not related, the workflow uses a general-purpose LLM to generate a response.
4.  **Document Retrieval:** The `retrieve` node uses Pinecone to find relevant documents based on the query embedding.
5.  **Relevance Evaluation:** The `evaluate` node filters out irrelevant documents using an LLM.
6.  **Response Generation:**
    * If relevant documents are found, the `respond` node generates a response using the retrieved context.
    * If no relevant documents are found, the `refine` node suggests a refined query, and the process repeats.
7.  **Output:** The generated response is displayed in the Streamlit app.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create `.env` File:**
    * Create a `.env` file in the root directory of the project.
    * Add your Pinecone and Groq API keys:
        ```
        PINECONE_API_KEY=your_pinecone_api_key
        GROQ_API_KEY=your_groq_api_key
        ```
5.  **Run the Scraper:**
    ```bash
    python Scraper.py
    ```
6.  **Run the clean text script**
    ```bash
    python clean_txt.py
    ```
7.  **Run the setup vector database script**
    ```bash
    python setup_vector_database.py
    ```
8.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

Now, you can access the application in your browser at the displayed URL.
