from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from typing import List
from app.analyzer import MoEAnalyzer, PromptAnalysisRequest, LoadModelRequest

PROMPTS_DIR = (Path(__file__).resolve().parent.parent / "prompts").resolve()
MODELS_DIR = (Path(__file__).resolve().parent.parent / "models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MODEL_EXTENSIONS = {
    ".json", ".safetensors", ".bin", ".pt", ".pth", ".txt",
    ".model", ".tiktoken", ".yaml", ".yml"
}
MAX_UPLOAD_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 Go max per file

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

@app.get("/api/model/info")
def get_model_info():
    return analyzer.get_model_info()

@app.get("/api/models/local")
def get_local_models():
    if not MODELS_DIR.exists():
        return []
    local_models = []
    for entry in sorted(MODELS_DIR.iterdir()):
        if entry.is_dir():
            files = [f.name for f in entry.glob("**/*") if f.is_file()]
            local_models.append({
                "id": str(entry.resolve()),
                "name": entry.name,
                "path": str(entry.resolve()),
                "files_count": len(files),
            })
    return local_models

@app.post("/api/models/upload")
def upload_model_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni")

    model_root_dir = None
    model_name = None

    for file in files:
        raw_name = file.filename or ""
        clean_name = raw_name.replace("\\", "/").strip()

        if not clean_name:
            raise HTTPException(status_code=400, detail="Nom de fichier vide ou invalide")

        # Rejeter tout chemin absolu ou lettre de lecteur Windows
        if clean_name.startswith("/") or ":" in clean_name:
            raise HTTPException(status_code=400, detail=f"Chemin absolu interdit: {clean_name}")

        # Séparer les segments
        parts = [p for p in clean_name.split("/") if p and p != "."]

        # Rejeter le path traversal
        if any(p == ".." for p in parts):
            raise HTTPException(status_code=400, detail=f"Path traversal ('..') interdit: {clean_name}")

        # Vérifier l'extension
        suffix = Path(parts[-1]).suffix.lower()
        if suffix not in ALLOWED_MODEL_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Extension non autorisée pour le fichier '{parts[-1]}': {suffix}. Extensions autorisées: {sorted(list(ALLOWED_MODEL_EXTENSIONS))}"
            )

        # Déterminer le dossier racine du modèle
        if len(parts) > 1:
            current_model_name = parts[0]
            current_model_dir = (MODELS_DIR / current_model_name).resolve()
        else:
            current_model_name = Path(parts[0]).stem
            current_model_dir = (MODELS_DIR / current_model_name).resolve()

        if model_root_dir is None:
            model_root_dir = current_model_dir
            model_name = current_model_name

        dest_path = (MODELS_DIR / Path(*parts)).resolve()

        # Sécurité : vérifier que dest_path reste sous MODELS_DIR
        try:
            dest_path.relative_to(MODELS_DIR)
        except ValueError:
            raise HTTPException(status_code=400, detail="Accès non autorisé en dehors du dossier models")

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        size = 0
        with open(dest_path, "wb") as f_out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_FILE_SIZE:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Fichier trop volumineux: {parts[-1]} (limite max: 2 Go)"
                    )
                f_out.write(chunk)

    return {
        "status": "success",
        "model_id": str(model_root_dir),
        "model_name": model_name,
        "model_path": str(model_root_dir),
        "files_count": len(files),
    }

