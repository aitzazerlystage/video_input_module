from fastapi import FastAPI
from pydantic import BaseModel, Field
from video_module_app import description_chain, QandA_chain, upload_video, make_vector_store
from typing import List
import uuid

app = FastAPI(
    title="Video Chatbot API",
    description="This API analyzes videos and lets you ask questions based on the video content.",
    version="1.0.0"
)

# ---------- Request Schemas ----------
class VideoRequest(BaseModel):
    file_paths: List[str] = Field(
        ...,
        description="List of absolute paths to video files on your system",
        example=[
            "C:\\Users\\Downloads\\video1.mp4",
            "C:\\Users\\Downloads\\video2.mp4"
        ]
    )


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        description="Your natural language question about the video",
        example="What is happening in the video?"
    )


# ---------- API Endpoints ----------
@app.post(
    "/analyze-video",
    summary="Analyze Video",
    description="Runs the description chain using the provided video path and generates a textual description."
)

@app.post("/analyze-video")
def analyze_video(req: VideoRequest):
    try:
        results = []
        userid = str(uuid.uuid4())[:8]

        try:
            make_vector_store(userid)
        except Exception as e:
            print(f"❌ Vector store creation failed: {e}")
            return {"error": str(e)}

        # Loop through each provided file path
        for file_path in req.file_paths:
            result = description_chain.invoke(file_path)
            results.append(result)

        return {"descriptions": results}
    except Exception as e:
        return {"error": str(e)}



@app.post(
    "/ask-question",
    summary="Ask Question",
    description="Runs the Q&A chain with a given question about the previously analyzed video."
)
def ask_question(req: QuestionRequest):
    """
    Takes a question as input and returns an answer based on the analyzed video.
    """
    try:
        result = QandA_chain.invoke(req.question)
        return {"answer": result}
    except Exception as e:
        return {"error": str(e)}
