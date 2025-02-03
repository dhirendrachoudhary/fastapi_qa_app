# FastAPI QA System with FAISS and Docker Deployment

## Overview
This is a **FastAPI-based** Question-Answering (QA) system that utilizes **FAISS** for vector search and **OpenAI** for answering user queries. The application is deployed using **Docker** and tested via **Swagger UI**.

---

## Features
- FastAPI-based **REST API** for question-answering
- **FAISS** for vector-based similarity search
- **HuggingFace Embeddings** for text representation
- **Dockerized Deployment**
- **Swagger UI** (`/docs`) for API testing

---

## Project Structure
```
/fastapi_qa_app
│── Dockerfile
│── requirements.txt
│── app.py
│── utils.py
│── .env
│── faiss_index/ (Pre-existing FAISS index folder)
│── models/ (If using additional models)
│── README.md
```

---

## Setup and Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/fastapi-qa-app.git
cd fastapi-qa-app
```

### **2️⃣ Create a Virtual Environment** (Optional)
```bash
python3 -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate    # For Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the FastAPI App Locally**
```bash
uvicorn app:app --reload
```

- Open your browser and go to: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Deploy Using Docker
### **1️⃣ Build the Docker Image**
```bash
docker build -t fastapi-qa-app .
```

### **2️⃣ Run the Docker Container**
```bash
docker run -d -p 8000:8000 --name fastapi_qa_container fastapi-qa-app
```

### **3️⃣ Verify the Running Container**
```bash
docker ps
```
Expected Output:
```
CONTAINER ID   IMAGE          COMMAND                  STATUS         PORTS                    NAMES
xyz12345678    fastapi-qa-app "uvicorn app:app --…"   Up 10 seconds  0.0.0.0:8000->8000/tcp   fastapi_qa_container
```

---

## Testing via Swagger UI
1. Open **Swagger UI**:
   - [http://localhost:8000/docs](http://localhost:8000/docs)
2. Click on **`POST /ask`** endpoint.
3. Enter a **plain text** question (e.g., `"What is LangChain?"`).
4. Click **"Execute"** and check the response.

#### **Expected Response**
```json
{
  "question": "What is LangChain?",
  "answer": "LangChain is a framework for building applications using large language models."
}
```

---

## Stopping & Removing Docker Containers
```bash
docker stop fastapi_qa_container
docker rm fastapi_qa_container
docker rmi fastapi-qa-app
```

---

## Notes
- Ensure **FAISS index** is pre-generated (`faiss_index/` directory exists).
- The `.env` file should contain **OpenAI API Key**:
  ```
  OPENAI_API_KEY=your-api-key-here
  ```
- Modify the `app.py` to include additional error handling and logging if required.

---

## Contribution
Pull requests are welcome! Feel free to submit issues for improvements.

---

## License
MIT License

