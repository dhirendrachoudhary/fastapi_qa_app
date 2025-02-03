from fastapi import FastAPI, HTTPException, Form
from typing import Any
import os
import logging
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from utils import create_knowledge_vector_database
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv(dotenv_path="../.env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Q&A Chatbot with Vector Store",
    description="Ask questions based on the ",
    version="1.0"
)

# Load embedding model
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True},
    )
except Exception as e:
    logger.error(f"Failed to load embedding model: {str(e)}")
    raise RuntimeError("Error initializing embedding model. Check model name and dependencies.")

# Create a knowledge vector database if not already created
FAISS_INDEX_PATH = "faiss_index"

if not os.path.exists(FAISS_INDEX_PATH):
    try:
        create_knowledge_vector_database(embedding_model, FAISS_INDEX_PATH)
    except Exception as e:
        logger.error(f"Error creating FAISS knowledge base: {str(e)}")
        raise RuntimeError("Failed to create knowledge vector database. Check input data.")

# Load FAISS knowledge vector database with exception handling
try:
    KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embedding_model, 
        distance_strategy=DistanceStrategy.COSINE, 
        allow_dangerous_deserialization=True
    )
except FileNotFoundError:
    logger.error(f"FAISS index file not found at {FAISS_INDEX_PATH}")
    raise RuntimeError(f"FAISS index file missing. Ensure {FAISS_INDEX_PATH} exists or recreate it.")
except Exception as e:
    logger.error(f"Error loading FAISS index: {str(e)}")
    raise RuntimeError("Failed to load knowledge vector database.")

# Function to answer questions with error handling
def answer_question(knowledge_base, user_question: str):
    try:
        # Retrieve relevant documents
        docs = knowledge_base.similarity_search(user_question)
        if not docs:
            return "No relevant information found in the knowledge base."

        # Generate response using OpenAI LLM
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)

        # Ensure formatted output with \n for readability
        return response.replace("\n", "\\n")

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating response from OpenAI API.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your request.")

# API Endpoint with Form Input (to allow plain text questions)
@app.post("/ask")
def get_answer(question: str = Form(...)) -> Any:
    """
    Accepts a direct text input question and returns an answer.
    """
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        answer = answer_question(KNOWLEDGE_VECTOR_DATABASE, question)
        return {"question": question, "answer": answer}

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Unhandled error in API: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Run the app using: uvicorn app:app --reload