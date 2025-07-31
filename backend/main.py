from fastapi import FastAPI, WebSocket, HTTPException
from schemas import PromptRequest, FeedbackRequest
from services import prompt_service, feedback_service, history_service

app = FastAPI()
#app = FastAPI(debug=True)

@app.post("/api/prompt")
async def prompt_endpoint(data: PromptRequest):
    return await prompt_service.handle_prompt(data)

@app.post("/api/feedback")
async def feedback_endpoint(data: FeedbackRequest):
    return feedback_service.handle_feedback(data)

@app.get("/api/history")
async def history_endpoint(id: int = None):
    return history_service.get_history(id)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Echo: {data}")
