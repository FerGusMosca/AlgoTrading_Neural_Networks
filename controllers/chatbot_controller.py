from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from logic_layer.chatbot_logic import ask_bot

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

@router.post("/ask")
async def ask_question(request: Request):
    try:
        payload = await request.json()
        question = payload.get("question")

        if not question:
            raise HTTPException(status_code=400, detail="Missing 'question' in request body")

        answer = ask_bot(question)
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
