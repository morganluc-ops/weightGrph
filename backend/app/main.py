from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from app.analyzer import MoEAnalyzer, PromptAnalysisRequest, LoadModelRequest

PROMPTS_DIR = (Path(__file__).resolve().parent.parent / "prompts").resolve()

app = FastAPI(title="MoE Weight & Expert Analyzer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = MoEAnalyzer()

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/domains")
def get_domains():
    return analyzer.get_available_domains()

@app.get("/api/models")
def get_models():
    return analyzer.get_available_models()

@app.get("/api/model/progress")
def get_model_progress():
    return analyzer.get_download_progress()

@app.post("/api/model/load")
def load_model(payload: LoadModelRequest):
    try:
        return analyzer.load_model(payload.model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/analyze/{domain_id}")
def analyze_domain(domain_id: str):
    try:
        return analyzer.analyze_domain(domain_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/analyze-prompt")
def analyze_prompt(payload: PromptAnalysisRequest):
    try:
        # Check if the prompt is an id of a prompt file
        if payload.prompt.endswith(".md"):
            # Extract only the file name to prevent path traversal
            file_name = Path(payload.prompt).name
            prompt_file = (PROMPTS_DIR / file_name).resolve()

            # Ensure prompt_file stays within PROMPTS_DIR
            try:
                prompt_file.relative_to(PROMPTS_DIR)
            except ValueError:
                raise HTTPException(status_code=400, detail="Accès non autorisé au fichier de prompt")

            if prompt_file.is_file():
                with open(prompt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                return analyzer.analyze_prompt(content)
            else:
                # If a .md file was explicitly specified but does not exist, reject or treat as invalid file
                raise HTTPException(status_code=400, detail=f"Fichier de prompt introuvable: {file_name}")

        return analyzer.analyze_prompt(payload.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/prompts")
def get_prompts():
    if not PROMPTS_DIR.exists():
        return []

    files = [f.name for f in PROMPTS_DIR.glob("*.md") if f.is_file()]
    return [{"id": f, "name": f.replace(".md", "")} for f in sorted(files)]

@app.post("/api/prompts/upload")
def upload_prompt(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .md sont autorisés")

    file_name = Path(file.filename).name
    if not file_name or file_name == ".md":
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    prompt_file = (PROMPTS_DIR / file_name).resolve()

    try:
        prompt_file.relative_to(PROMPTS_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Accès non autorisé au fichier de prompt")

    try:
        content = file.file.read()
        with open(prompt_file, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du fichier: {str(e)}")

    return {"id": file_name, "name": file_name.replace(".md", "")}

@app.get("/api/model/info")
def get_model_info():
    return analyzer.get_model_info()
