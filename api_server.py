from fastapi import FastAPI
from pydantic import BaseModel, Field
from video_module_app import description_chain, QandA_chain

app = FastAPI(
    title="Video Chatbot API",
    description="This API analyzes videos and lets you ask questions based on the video content.",
    version="1.0.0"
)

# ---------- Request Schemas ----------
class VideoRequest(BaseModel):
    file_path: str = Field(
        ...,
        description="Absolute path to the video file on your system",
        example="C:\\Users\\Downloads\\video.mp4"
    )


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        description="Your natural language question about the video",
        example="What is happening in the first 30 seconds of the video?"
    )


# ---------- API Endpoints ----------
@app.post(
    "/analyze-video",
    summary="Analyze Video",
    description="Runs the description chain using the provided video path and generates a textual description."
)
def analyze_video(req: VideoRequest):
    """
    Takes a video path as input and returns a description of the video.
    """
    try:
        result = description_chain.invoke(req.file_path)
        return {"description": result}
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
