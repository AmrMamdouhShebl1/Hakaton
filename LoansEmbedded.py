import os
import re
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Check for Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in the .env file.")

# Initialize Google embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Regular expressions for product parsing
PRODUCT_PATTERN = r"===PRODUCT_START===\s+(.*?)===PRODUCT_END==="
FIELD_PATTERN = r"([A-Z_]+):\s*(.*?)(?=\n[A-Z_]+:|$)"

# Text splitter for document chunks
text_splitter = CharacterTextSplitter(
    separator="===PRODUCT_END===",
    chunk_size=2000,
    chunk_overlap=200
)

# List of Word documents containing loan products info
docx_files = [
    "docs/english/loan.docx",
    # Add more documents as needed
]


# Function to parse product data from documents
def process_product_documents(file_paths):
    """
    Process Word documents containing financial product listings.
    """
    all_docs = []

    for file_path in file_paths:
        try:
            print(f"Processing {file_path}...")

            # Load the document
            docx_loader = Docx2txtLoader(file_path)
            raw_docs = docx_loader.load()

            # Process each document
            for doc in raw_docs:
                text = doc.page_content

                # Find all product blocks
                product_blocks = re.findall(PRODUCT_PATTERN, text, re.DOTALL)

                # Process each product block
                for i, block in enumerate(product_blocks, 1):
                    print(f"  Found product {i} in {file_path}")

                    # Extract fields for each product
                    fields = re.findall(FIELD_PATTERN, block, re.DOTALL)
                    product_data = {key.lower(): value.strip() for key, value in fields}

                    # Create metadata for the document
                    metadata = {
                        "source": file_path,
                        "product_id": product_data.get("id", f"unknown-{i}"),
                        "product_type": product_data.get("type", "unknown"),
                        "product_title": product_data.get("title", "unknown"),
                    }

                    # Create content from all product fields
                    content = "\n".join([f"{k}: {v}" for k, v in product_data.items()])

                    # Create a document for this product
                    product_doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(product_doc)

        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")

    return all_docs


# Process the documents
product_docs = process_product_documents(docx_files)

# Print summary
print(f"\nFound {len(product_docs)} loan products in documents")

# Check if we have any documents to embed
if not product_docs:
    print("No products found to embed. Check your documents and format.")
else:
    # Create the vector store
    print("\nCreating vector store with Google embeddings...")
    db = Chroma.from_documents(
        product_docs,
        embedding=embeddings,
        persist_directory="emb_loans"  # Directory for loan products
    )

    print(f"Successfully created loans vector store in 'emb_loans'")
    print(f"Embedded {len(product_docs)} loan products in total.")

print("\nLoan products embedding process complete!")