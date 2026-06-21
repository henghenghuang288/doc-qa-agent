import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import answer_question
from .document_processor import DocumentError, chunk_text, parse_document
from .session_store import create_session, get_session

app = FastAPI(title="企业文档问答 Agent", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    live = bool(
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    return {"status": "ok", "live_mode": live}


@app.post("/api/session")
def new_session():
    session = create_session()
    return {"session_id": session.session_id}


@app.post("/api/session/{session_id}/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期,请刷新页面重新开始")

    raw = await file.read()
    try:
        parsed = parse_document(file.filename, raw)
    except DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_text(parsed.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档解析后未能切分出有效内容")

    session.index.add_documents(chunks, source=parsed.filename)
    session.filenames.append(parsed.filename)

    return {
        "filename": parsed.filename,
        "char_count": parsed.char_count,
        "chunk_count": len(chunks),
        "total_files": len(session.filenames),
        "total_chunks": len(session.index.chunks),
    }


class Question(BaseModel):
    question: str


@app.post("/api/session/{session_id}/ask")
def ask(session_id: str, body: Question):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期,请刷新页面重新开始")
    if not session.index.chunks:
        raise HTTPException(status_code=400, detail="当前会话还没有上传任何文档")

    return answer_question(session.index, body.question)


_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
