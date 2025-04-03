import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from pinecone import ServerlessSpec
import itertools
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Define index parameters
index_name = "web-content-index"
dimension = 384  # Dimension of the embedding model
metric = "cosine"
cloud = "aws"
region = "us-east-1"

# Check if the index exists
existing_indexes = [index_info['name'] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region)
    )

# Connect to the index
index = pc.Index(index_name)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load cleaned text
with open("occams_clean_text.txt", "r", encoding="utf-8") as f:
    occams_clean_text = f.read()

# Use LangChain for overlapping chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)
chunks = text_splitter.split_text(occams_clean_text)

# Function to create batches of a specified size
def batch(iterable, n=1):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch

# Prepare and upload data to Pinecone in batches
batch_size = 100  # Recommended batch size
chunk_id = 0  # Ensure unique chunk IDs
with ThreadPoolExecutor() as executor:
    futures = []
    for chunk_batch in batch(chunks, batch_size):
        vectors = []
        for text in chunk_batch:
            embedding = model.encode(text).tolist()
            metadata = {"text": text[:]}  # Store only the first 200 characters
            vectors.append((f"chunk_{chunk_id}", embedding, metadata))
            chunk_id += 1  # Ensure unique chunk ID
        
        # Asynchronously upload batch to Pinecone
        futures.append(executor.submit(index.upsert, vectors))

    # Wait for all futures to complete
    for future in futures:
        future.result()

print("Data successfully stored in Pinecone!")

