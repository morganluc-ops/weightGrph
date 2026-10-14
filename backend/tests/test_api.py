import pytest
from fastapi.testclient import TestClient
from app.main import app
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

def test_get_local_models_endpoint():
    response = client.get("/api/models/local")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_upload_model_valid():
    import shutil
    from pathlib import Path
    from app.main import MODELS_DIR

    test_model_dir = MODELS_DIR / "my-moe-test"
    try:
        files = [
            ("files", ("my-moe-test/config.json", b'{"model_type": "moe"}', "application/json")),
            ("files", ("my-moe-test/model.safetensors", b'fake-tensor-bytes', "application/octet-stream")),
        ]
        response = client.post("/api/models/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_name"] == "my-moe-test"
        assert data["files_count"] == 2
        assert "my-moe-test" in data["model_path"]

        # Verify it appears in local models
        list_res = client.get("/api/models/local")
        assert list_res.status_code == 200
        local_names = [m["name"] for m in list_res.json()]
        assert "my-moe-test" in local_names
    finally:
        if test_model_dir.exists():
            shutil.rmtree(test_model_dir, ignore_errors=True)


def test_upload_model_path_traversal_rejected():
    # Attempting traversal with ../
    files = [
        ("files", ("../../evil.json", b'{"bad": true}', "application/json")),
    ]
    response = client.post("/api/models/upload", files=files)
    assert response.status_code == 400
    assert "Path traversal" in response.json()["detail"] or "interdit" in response.json()["detail"]

    # Attempting traversal with backslash ..\
    files_bs = [
        ("files", ("..\\evil_bs.json", b'{"bad": true}', "application/json")),
    ]
    response_bs = client.post("/api/models/upload", files=files_bs)
    assert response_bs.status_code == 400
    assert "Path traversal" in response_bs.json()["detail"] or "interdit" in response_bs.json()["detail"]

def test_upload_model_absolute_path_rejected():
    files = [
        ("files", ("/etc/config.json", b'{"bad": true}', "application/json")),
    ]
    response = client.post("/api/models/upload", files=files)
    assert response.status_code == 400
    assert "absolu interdit" in response.json()["detail"]

def test_upload_model_disallowed_extension_rejected():
    files = [
        ("files", ("bad-model/malicious.exe", b'MZ...', "application/octet-stream")),
    ]
    response = client.post("/api/models/upload", files=files)
    assert response.status_code == 400
    assert "Extension non autorisée" in response.json()["detail"]

def test_upload_model_pickle_extensions_rejected():
    # .bin, .pt, and .pth must be strictly rejected due to pickle remote code execution risks
    for ext in [".bin", ".pt", ".pth"]:
        files = [
            ("files", (f"model-pickle/weights{ext}", b'pickle-bytes', "application/octet-stream")),
        ]
        response = client.post("/api/models/upload", files=files)
        assert response.status_code == 400
        assert "Extension non autorisée" in response.json()["detail"]
        assert ".safetensors" in response.json()["detail"]

def test_upload_model_duplicate_conflict_409():
    import shutil
    from app.main import MODELS_DIR

    model_dir = MODELS_DIR / "dup-test-moe"
    try:
        files = [
            ("files", ("dup-test-moe/config.json", b'{"model_type": "moe"}', "application/json")),
            ("files", ("dup-test-moe/model.safetensors", b'fake-weights', "application/octet-stream")),
        ]
        # First upload: success
        res1 = client.post("/api/models/upload", files=files)
        assert res1.status_code == 200

        # Second upload with same model directory: 409 Conflict
        res2 = client.post("/api/models/upload", files=files)
        assert res2.status_code == 409
        assert "existe déjà" in res2.json()["detail"]
    finally:
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)

def test_upload_model_multiple_roots_rejected_400():
    files = [
        ("files", ("model-alpha/config.json", b'{"model_type": "moe"}', "application/json")),
        ("files", ("model-beta/config.json", b'{"model_type": "moe"}', "application/json")),
    ]
    response = client.post("/api/models/upload", files=files)
    assert response.status_code == 400
    assert "même dossier racine" in response.json()["detail"]

def test_upload_model_atomic_validation_no_residue():
    from app.main import MODELS_DIR

    target_dir = MODELS_DIR / "failing-model"
    # Second file has an invalid extension (.bin), so validation fails before writing anything
    files = [
        ("files", ("failing-model/config.json", b'{"model_type": "moe"}', "application/json")),
        ("files", ("failing-model/weights.bin", b'evil', "application/octet-stream")),
    ]
    response = client.post("/api/models/upload", files=files)
    assert response.status_code == 400
    assert not target_dir.exists()

def test_upload_model_payload_too_large_413(monkeypatch):
    import app.main as main_mod
    from app.main import MODELS_DIR

    # Temporarily set max file size to 10 bytes to trigger 413
    monkeypatch.setattr(main_mod, "MAX_UPLOAD_FILE_SIZE", 10)

    target_dir = MODELS_DIR / "large-model"
    try:
        files = [
            ("files", ("large-model/config.json", b'12345678901234567890', "application/json")),
        ]
        response = client.post("/api/models/upload", files=files)
        assert response.status_code == 413
        assert "trop volumineux" in response.json()["detail"]
        # Ensure rollback was performed and no directory remains
        assert not target_dir.exists()
    finally:
        if target_dir.exists():
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)



