import os
import pandas as pd
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.docstore.document import Document
import re

# Load environment variables
load_dotenv()

# Check for Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in the .env file.")

# Initialize Google Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Regular expressions for property parsing
PROPERTY_PATTERN = r"===PROPERTY_START===\s+(.*?)===PROPERTY_END==="
FIELD_PATTERN = r"([A-Z]+):\s*(.*?)(?=\n[A-Z]+:|$)"

# Text splitter for document chunks
text_splitter = CharacterTextSplitter(
    separator="===PROPERTY_END===",
    chunk_size=2000,
    chunk_overlap=200
)

# List of Arabic Word documents to embed
docx_files = [
    "docs/arabic/properties.docx",
    # Add more Arabic property documents as needed
]


# Function to parse property data from documents
def process_property_documents(file_paths):
    """
    Process Word documents containing property listings.
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

                # Find all property blocks
                property_blocks = re.findall(PROPERTY_PATTERN, text, re.DOTALL)

                # Process each property block
                for i, block in enumerate(property_blocks, 1):
                    print(f"  Found property {i} in {file_path}")

                    # Extract fields for each property
                    fields = re.findall(FIELD_PATTERN, block, re.DOTALL)
                    property_data = {key.lower(): value.strip() for key, value in fields}

                    # Create metadata for the document
                    metadata = {
                        "source": file_path,
                        "property_id": property_data.get("id", f"unknown-{i}"),
                        "property_type": property_data.get("type", "unknown"),
                        "property_status": property_data.get("status", "unknown"),
                        "property_location": property_data.get("location", "unknown"),
                        "property_price": property_data.get("price", "unknown"),
                    }

                    # Create content from all property fields
                    content = "\n".join([f"{k}: {v}" for k, v in property_data.items()])

                    # Create a document for this property
                    property_doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(property_doc)

        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")

    return all_docs


# Process the documents
property_docs = process_property_documents(docx_files)

# Print summary
print(f"\nFound {len(property_docs)} properties in Arabic documents")

# Check if we have any documents to embed
if not property_docs:
    print("No properties found to embed. Check your documents and format.")
else:
    # Create the vector store
    print("\nCreating vector store with Google embeddings...")
    db = Chroma.from_documents(
        property_docs,
        embedding=embeddings,
        persist_directory="emb_default_arabic"  # Specify directory to persist embeddings
    )

    print(f"Successfully created Arabic vector store in 'emb_default_arabic'")
    print(f"Embedded {len(property_docs)} Arabic properties in total.")

print("\nArabic embedding process complete!")