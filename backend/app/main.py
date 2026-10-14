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
        if payload.prompt.lower().endswith(".md"):
            # Extract only the file name to prevent path traversal
            file_name = Path(payload.prompt).name
            prompt_file = (PROMPTS_DIR / file_name).resolve()

            # Ensure prompt_file stays within PROMPTS_DIR
            try:
                prompt_file.relative_to(PROMPTS_DIR)
            except ValueError:
                raise HTTPException(status_code=400, detail="Accès non autorisé au fichier de prompt")

            # Fallback to normalized lowercase .md if exact casing does not exist on disk (Linux)
            if not prompt_file.is_file() and file_name.lower().endswith(".md"):
                normalized_file = (PROMPTS_DIR / f"{file_name[:-3]}.md").resolve()
                if normalized_file.is_file():
                    prompt_file = normalized_file

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

MAX_PROMPT_FILE_SIZE = 1 * 1024 * 1024  # 1 Mo
CHUNK_SIZE = 64 * 1024  # 64 Ko

@app.post("/api/prompts/upload")
def upload_prompt(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .md sont autorisés")

    raw_name = Path(file.filename).name
    if not raw_name or raw_name.lower() == ".md":
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")

    # Normaliser systématiquement l'extension en .md minuscule
    stem = raw_name[:-3] if raw_name.lower().endswith(".md") else raw_name
    file_name = f"{stem}.md"

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    prompt_file = (PROMPTS_DIR / file_name).resolve()

    try:
        prompt_file.relative_to(PROMPTS_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Accès non autorisé au fichier de prompt")

    if prompt_file.exists():
        raise HTTPException(status_code=409, detail=f"Un prompt nommé '{file_name}' existe déjà")

    chunks = []
    total_size = 0
    try:
        while True:
            chunk = file.file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_PROMPT_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="Le fichier dépasse la taille maximale autorisée (1 Mo)"
                )
            chunks.append(chunk)

        content = b"".join(chunks)

        try:
            _ = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Le contenu du fichier n'est pas un texte UTF-8 valide"
            )

        with open(prompt_file, "xb") as f:
            f.write(content)
    except FileExistsError:
        raise HTTPException(status_code=409, detail=f"Un prompt nommé '{file_name}' existe déjà")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du fichier: {str(e)}")

    return {"id": file_name, "name": stem}

@app.get("/api/model/info")
def get_model_info():
    return analyzer.get_model_info()
