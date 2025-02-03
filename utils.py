import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import openai


def create_chunks():
    # load all the chunks and cretae a dic of chunks and their filename
    chunks = {}
    direcory = "/Users/dhirendrachoudhary/Desktop/Workstation/AbInBev/AbinbevGenAIAssignment/demo_bot_data/ubuntu-docs"
    for root, dirs, files in os.walk(direcory):
        # Skip hidden directories (starting with ".")
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file.endswith(".md") and not file.startswith("."):  # Skip hidden files
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                chunks[file_path.split('/')[-1]] = content
    return chunks

def load_env_vas():
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")

def create_knowledge_vector_database(embedding_model, knowledge_base_path):
    """
    Create a knowledge vector database from a knowledge base and an embedding model.
    """
    if not os.path.exists(knowledge_base_path):
        load_dotenv(dotenv_path="../.env")
        load_env_vas()

        chunks = create_chunks()
        
        RAW_KNOWLEDGE_BASE = [
            LangchainDocument(page_content=doc, metadata={"source": key}) for key, doc in chunks.items()
        ]

        MARKDOWN_SEPARATORS = [
            "\n#{1,6} ",
            "```\n",
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            "\n\n",
            "\n",
            " ",
            "",
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # The maximum number of characters in a chunk: we selected this value arbitrarily
            chunk_overlap=100,  # The number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=MARKDOWN_SEPARATORS,
        )

        docs_processed = []
        for doc in RAW_KNOWLEDGE_BASE:
            docs_processed += text_splitter.split_documents([doc])

        KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        KNOWLEDGE_VECTOR_DATABASE.save_local(knowledge_base_path)