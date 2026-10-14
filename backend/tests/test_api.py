import io
import pytest
from fastapi.testclient import TestClient
from app.main import app, PROMPTS_DIR
from app.analyzer import DEFAULT_REAL_MOE_MODEL

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_domains():
    response = client.get("/api/domains")
    assert response.status_code == 200
    domains = response.json()
    assert "philosophie" in domains
    assert "mathematiques" in domains

def test_analyze_domain():
    response = client.get("/api/analyze/philosophie")
    assert response.status_code == 200
    data = response.json()
    assert data["domain_id"] == "philosophie"
    assert data["is_real_inference"] is True
    assert "experts" in data
    assert len(data["experts"]) > 0

    experts = data["experts"]
    for i in range(len(experts) - 1):
        assert experts[i]["usage_percentage"] >= experts[i + 1]["usage_percentage"]

def test_analyze_invalid_domain():
    response = client.get("/api/analyze/invalid_domain")
    assert response.status_code == 404
    assert response.json()["detail"] == "Domaine inconnu: invalid_domain"

def test_get_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    info = response.json()
    assert info["is_real_model"] is True
    assert info["num_layers"] == 4
    assert info["experts_per_layer"] == 8

def test_analyze_prompt_endpoint():
    response = client.post("/api/analyze-prompt", json={"prompt": "Écris un algorithme Python pour calculer une intégrale"})
    assert response.status_code == 200
    data = response.json()
    assert data["domain_id"] == "custom_prompt"
    assert data["is_real_inference"] is True
    assert len(data["experts"]) > 0
    assert len(data["token_routing"]) > 0

def test_analyze_empty_prompt_endpoint():
    response = client.post("/api/analyze-prompt", json={"prompt": ""})
    assert response.status_code == 400
    assert "Le prompt ne peut pas être vide" in response.json()["detail"]

def test_get_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert all(m["is_real"] is True for m in models)

def test_get_model_progress_endpoint():
    response = client.get("/api/model/progress")
    assert response.status_code == 200
    prog = response.json()
    assert "status" in prog
    assert "progress_percent" in prog

def test_load_real_model_endpoint():
    response = client.post("/api/model/load", json={"model_id": DEFAULT_REAL_MOE_MODEL})
    assert response.status_code == 200
    assert response.json()["model_id"] == DEFAULT_REAL_MOE_MODEL
    assert response.json()["is_real_model"] is True

def test_get_prompts():
    response = client.get("/api/prompts")
    assert response.status_code == 200
    prompts = response.json()
    assert isinstance(prompts, list)
    assert len(prompts) >= 2
    assert any(p["id"] == "philosophie.md" for p in prompts)
    assert any(p["id"] == "code.md" for p in prompts)

def test_analyze_prompt_with_md_file():
    response = client.post("/api/analyze-prompt", json={"prompt": "philosophie.md"})
    assert response.status_code == 200
    data = response.json()
    assert data["domain_id"] == "custom_prompt"
    assert len(data["experts"]) > 0

def test_analyze_prompt_path_traversal_rejected():
    response = client.post("/api/analyze-prompt", json={"prompt": "../../etc/passwd.md"})
    assert response.status_code == 400
    assert "Fichier de prompt introuvable" in response.json()["detail"] or "Accès non autorisé" in response.json()["detail"]

def test_upload_prompt_success():
    filename = "test_custom_upload.md"
    file_path = PROMPTS_DIR / filename
    if file_path.exists():
        file_path.unlink()

    try:
        content = b"# Mon super prompt d'analyse\nExplique le fonctionnement des MoE."
        response = client.post(
            "/api/prompts/upload",
            files={"file": (filename, io.BytesIO(content), "text/markdown")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == filename
        assert data["name"] == "test_custom_upload"
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == content.decode("utf-8")

        # Vérifier qu'il est maintenant dans la liste
        list_res = client.get("/api/prompts")
        assert any(p["id"] == filename for p in list_res.json())

        # Vérifier qu'on peut l'analyser
        analyze_res = client.post("/api/analyze-prompt", json={"prompt": filename})
        assert analyze_res.status_code == 200
    finally:
        if file_path.exists():
            file_path.unlink()

def test_upload_prompt_invalid_extension():
    response = client.post(
        "/api/prompts/upload",
        files={"file": ("script.py", io.BytesIO(b"print('hello')"), "text/x-python")},
    )
    assert response.status_code == 400
    assert "Seuls les fichiers .md sont autorisés" in response.json()["detail"]

def test_upload_prompt_empty_filename():
    response = client.post(
        "/api/prompts/upload",
        files={"file": (".md", io.BytesIO(b"content"), "text/markdown")},
    )
    assert response.status_code == 400
    assert "Nom de fichier invalide" in response.json()["detail"]

def test_upload_prompt_sanitizes_path_traversal():
    filename = "test_traversal_clean.md"
    file_path = PROMPTS_DIR / filename
    if file_path.exists():
        file_path.unlink()

    try:
        traversal_name = f"../../sub/{filename}"
        response = client.post(
            "/api/prompts/upload",
            files={"file": (traversal_name, io.BytesIO(b"# Traversal"), "text/markdown")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == filename
        assert file_path.exists()
    finally:
        if file_path.exists():
            file_path.unlink()


