# 🎥 Video Chatbot API

This project is an **AI-powered video analysis chatbot** built with **LangChain**, **Google Gemini (Generative AI)**, **OpenAI embeddings**, and **Pinecone**.  
It lets you **upload a video**, generate a detailed **description**, store embeddings in Pinecone, and then **ask natural language questions** about the video content via a FastAPI server.

---

## 🚀 Features
- 📤 **Video Upload & Analysis** – Upload a video and generate a detailed textual description using **Google Gemini**.  
- 🧠 **Vector Store with Pinecone** – Video descriptions are split into chunks and stored as embeddings for semantic retrieval.  
- ❓ **Question Answering** – Ask questions about the video and get context-aware answers.  
- ⚡ **FastAPI Server** – REST API endpoints for analyzing videos and querying them.  

---

## 📂 Project Structure
```
.
├── api_server.py         # FastAPI server (API endpoints)
├── video_module_app.py   # Core logic: LLMs, embeddings, Pinecone integration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Installation

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux / macOS
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables
Create a `.env` file in the root directory with the following:

```ini
# API Keys
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_pinecone_index_name
```

⚠️ **Note:** Make sure to add your API keys and configuration values in the `.env` file before running the project.

---

## ▶️ Running the API

Start the FastAPI server with:
```bash
uvicorn api_server:app --reload
```

The server will run at:
```
http://127.0.0.1:8000
```

---

## 📌 API Endpoints

### 1. **Analyze Video**
**POST** `/analyze-video`  
Uploads and analyzes a video, storing its description in Pinecone.

**Request body:**
```json
{
  "file_path": "C:\\Users\\Downloads\\video.mp4"
}
```

**Response:**
```json
{
  "description": "A man is walking across the street..."
}
```

---

### 2. **Ask Question**
**POST** `/ask-question`  
Ask a natural language question about the previously analyzed video.

**Request body:**
```json
{
  "question": "What happens in the first 30 seconds?"
}
```

**Response:**
```json
{
  "answer": "The video shows a man entering the building."
}
```

---

## 📦 Dependencies
Main libraries used:
- `fastapi` – API framework  
- `pydantic` – Data validation  
- `langchain` – Orchestration framework  
- `langchain-google-genai` – Google Gemini LLM integration  
- `langchain-openai` – OpenAI LLM + embeddings  
- `langchain-pinecone` – Pinecone vector store integration  
- `pinecone-client` – Pinecone client SDK  
- `google-generativeai` – Google GenAI SDK  
- `python-dotenv` – Environment variable management  
- `uvicorn` – ASGI server  

---

## 🧩 Example Workflow
1. **Upload & Analyze a Video** using `/analyze-video`  
2. The description is **split into chunks** and stored in **Pinecone** as embeddings.  
3. Ask a question via `/ask-question`.  
4. The system retrieves relevant context from Pinecone and queries the **LLM**.  

