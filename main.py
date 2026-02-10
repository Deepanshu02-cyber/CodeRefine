from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ai_engine import review_code
from fastapi.staticfiles import StaticFiles
from typing import Optional
from google import genai
import re

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class CodeRequest(BaseModel):
    code: str
    pro_prompt: str | None = ""


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/review")
async def review(req: CodeRequest):
    try:
        result = review_code(req.code, req.pro_prompt or "")
        if result.get("ok"):
            return result
        return JSONResponse(status_code=400, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
