from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from typing import List
from app.analyzer import (
    MoEAnalyzer,
    PromptAnalysisRequest,
    LoadModelRequest,
    MAX_PROMPT_LENGTH,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = (Path(__file__).resolve().parent.parent / "prompts").resolve()
MODELS_DIR = (Path(__file__).resolve().parent.parent / "models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

def get_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS")
    if raw is not None:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        return origins if origins else DEFAULT_CORS_ORIGINS
    return DEFAULT_CORS_ORIGINS

cors_origins = get_cors_origins()
allow_credentials = "*" not in cors_origins

ALLOWED_MODEL_EXTENSIONS = {
    ".safetensors",
    ".json",
    ".txt",
    ".model",
    ".tiktoken",
    ".yaml",
    ".yml",
}
MAX_UPLOAD_FILES_COUNT = 500
MAX_UPLOAD_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 Go max per file
MAX_UPLOAD_TOTAL_SIZE = 10 * 1024 * 1024 * 1024  # 10 Go max total per batch

app = FastAPI(title="MoE Weight & Expert Analyzer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
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
        logger.warning("Erreur lors du chargement du modèle '%s': %s", payload.model_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Erreur inattendue lors du chargement du modèle '%s': %s", payload.model_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne lors du chargement du modèle: {str(e)}")


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
                if len(content) > MAX_PROMPT_LENGTH:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Le contenu du fichier de prompt dépasse la limite autorisée ({MAX_PROMPT_LENGTH} caractères)",
                    )
                return analyzer.analyze_prompt(content)
            else:
                # If a .md file was explicitly specified but does not exist, reject or treat as invalid file
                raise HTTPException(status_code=400, detail=f"Fichier de prompt introuvable: {file_name}")

        if len(payload.prompt) > MAX_PROMPT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Le prompt dépasse la limite autorisée ({MAX_PROMPT_LENGTH} caractères)",
            )

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

    if len(files) > MAX_UPLOAD_FILES_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de fichiers trop élevé: {len(files)} (maximum autorisé: {MAX_UPLOAD_FILES_COUNT})"
        )

    # 1. Validation préalable complète (tout-ou-rien) de TOUS les fichiers avant la moindre écriture
    validated_files: List[tuple[UploadFile, List[str], Path]] = []
    common_root_name = None

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
        if not parts:
            raise HTTPException(status_code=400, detail="Chemin de fichier vide")

        # Rejeter le path traversal
        if any(p == ".." for p in parts):
            raise HTTPException(status_code=400, detail=f"Path traversal ('..') interdit: {clean_name}")

        # Vérifier l'extension (seul .safetensors et configs autorisés, pickle .bin/.pt/.pth strictement interdit)
        suffix = Path(parts[-1]).suffix.lower()
        if suffix not in ALLOWED_MODEL_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Extension non autorisée pour '{parts[-1]}': '{suffix}'. "
                    f"Seul le format sécurisé .safetensors et les configurations sont autorisés (.bin/.pt/.pth interdits). "
                    f"Extensions autorisées: {sorted(list(ALLOWED_MODEL_EXTENSIONS))}"
                )
            )

        # Déterminer et valider la racine commune (un seul dossier racine par envoi)
        root_name = parts[0] if len(parts) > 1 else Path(parts[0]).stem
        if common_root_name is None:
            common_root_name = root_name
        elif root_name != common_root_name:
            raise HTTPException(
                status_code=400,
                detail=f"Tous les fichiers doivent appartenir à un même dossier racine de modèle. Racines détectées: '{common_root_name}' et '{root_name}'"
            )

        dest_path = (MODELS_DIR / Path(*parts)).resolve()

        # Sécurité : vérifier que dest_path reste sous MODELS_DIR
        try:
            dest_path.relative_to(MODELS_DIR)
        except ValueError:
            raise HTTPException(status_code=400, detail="Accès non autorisé en dehors du dossier models")

        validated_files.append((file, parts, dest_path))

    # 2. Vérifier si le modèle existe déjà -> 409 Conflict
    target_model_dir = (MODELS_DIR / common_root_name).resolve()
    if target_model_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Le modèle '{common_root_name}' existe déjà sur le serveur"
        )

    # 3. Écriture atomique avec rollback complet en cas d'erreur
    total_size = 0
    created_dir = False
    try:
        target_model_dir.mkdir(parents=True, exist_ok=False)
        created_dir = True

        for file, parts, dest_path in validated_files:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            file_size = 0
            with open(dest_path, "wb") as f_out:
                while chunk := file.file.read(1024 * 1024):
                    chunk_len = len(chunk)
                    file_size += chunk_len
                    total_size += chunk_len

                    if file_size > MAX_UPLOAD_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Fichier trop volumineux: {parts[-1]} (taille max: 2 Go)"
                        )
                    if total_size > MAX_UPLOAD_TOTAL_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Taille totale du lot trop volumineuse (taille max cumulée: 10 Go)"
                        )
                    f_out.write(chunk)
    except Exception:
        # En cas d'erreur à n'importe quel moment de l'écriture : rollback complet
        if created_dir and target_model_dir.exists():
            import shutil
            shutil.rmtree(target_model_dir, ignore_errors=True)
        raise

    return {
        "status": "success",
        "model_id": str(target_model_dir),
        "model_name": common_root_name,
        "model_path": str(target_model_dir),
        "files_count": len(validated_files),
    }


