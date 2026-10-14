import io
from unittest.mock import patch
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
    assert "balance_metrics" in data
    bm = data["balance_metrics"]
    assert "entropy" in bm
    assert "gini_coefficient" in bm
    assert "dead_experts_count" in bm
    assert "total_experts" in bm
    assert "max_expert_load_share" in bm
    assert "layers_metrics" in bm
    assert len(bm["layers_metrics"]) == 4

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
    assert "balance_metrics" in data
    assert data["balance_metrics"]["total_experts"] == 32
    assert 0.0 <= data["balance_metrics"]["entropy"] <= 1.0

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

def test_upload_prompt_case_insensitive_extension():
    filename = "test_upper.MD"
    normalized_name = "test_upper.md"
    file_path = PROMPTS_DIR / normalized_name
    if file_path.exists():
        file_path.unlink()

    try:
        content = b"# Upper extension test\nEcris une fonction mathematique."
        response = client.post(
            "/api/prompts/upload",
            files={"file": (filename, io.BytesIO(content), "text/markdown")},
        )
        assert response.status_code == 200
        assert response.json()["id"] == normalized_name
        assert response.json()["name"] == "test_upper"
        assert file_path.exists()

        # Tester l'analyse du prompt uploadé avec le nom normalisé
        analyze_res1 = client.post("/api/analyze-prompt", json={"prompt": normalized_name})
        assert analyze_res1.status_code == 200
        assert analyze_res1.json()["domain_id"] == "custom_prompt"
        assert len(analyze_res1.json()["experts"]) > 0

        # Tester l'analyse du prompt avec l'extension .MD originale insensible à la casse
        analyze_res2 = client.post("/api/analyze-prompt", json={"prompt": filename})
        assert analyze_res2.status_code == 200
        assert analyze_res2.json()["domain_id"] == "custom_prompt"
    finally:
        if file_path.exists():
            file_path.unlink()

def test_upload_prompt_conflict_duplicate():
    # 'code.md' exists in PROMPTS_DIR
    response = client.post(
        "/api/prompts/upload",
        files={"file": ("code.md", io.BytesIO(b"# Duplicate code"), "text/markdown")},
    )
    assert response.status_code == 409
    assert "existe déjà" in response.json()["detail"]

def test_upload_prompt_race_condition_file_exists_error():
    filename = "race_test.md"
    with patch("builtins.open", side_effect=FileExistsError):
        response = client.post(
            "/api/prompts/upload",
            files={"file": (filename, io.BytesIO(b"# Content"), "text/markdown")},
        )
        assert response.status_code == 409
        assert "existe déjà" in response.json()["detail"]

def test_upload_prompt_payload_too_large():
    filename = "test_oversized.md"
    # Create content > 1 Mo (e.g. 1 Mo + 1024 bytes)
    oversized_content = b"A" * (1 * 1024 * 1024 + 1024)
    response = client.post(
        "/api/prompts/upload",
        files={"file": (filename, io.BytesIO(oversized_content), "text/markdown")},
    )
    assert response.status_code == 413
    assert "dépassé" in response.json()["detail"] or "1 Mo" in response.json()["detail"]

def test_upload_prompt_invalid_utf8():
    filename = "test_invalid_utf8.md"
    invalid_content = b"\x80\x81\xff\xfe"
    response = client.post(
        "/api/prompts/upload",
        files={"file": (filename, io.BytesIO(invalid_content), "text/markdown")},
    )
    assert response.status_code == 400
    assert "UTF-8 valide" in response.json()["detail"]


def test_analyze_prompt_oversized_text_rejected_400():
    oversized = "A" * 10_001
    response = client.post("/api/analyze-prompt", json={"prompt": oversized})
    assert response.status_code == 400
    assert "dépasse la limite autorisée" in response.json()["detail"] or "10000" in response.json()["detail"]


def test_analyze_prompt_oversized_file_rejected_400():
    filename = "temp_oversized_prompt.md"
    file_path = PROMPTS_DIR / filename
    try:
        file_path.write_text("A" * 10_001, encoding="utf-8")
        response = client.post("/api/analyze-prompt", json={"prompt": filename})
        assert response.status_code == 400
        assert "dépasse la limite autorisée" in response.json()["detail"]
    finally:
        if file_path.exists():
            file_path.unlink()


def test_load_model_path_traversal_rejected_400():
    response = client.post("/api/model/load", json={"model_id": "../../etc/passwd"})
    assert response.status_code == 400
    assert "path traversal" in response.json()["detail"] or "invalide" in response.json()["detail"]


def test_load_model_unauthorized_path_rejected_400():
    response = client.post("/api/model/load", json={"model_id": "/etc/shadow"})
    assert response.status_code == 400
    assert "non autorisé ou introuvable" in response.json()["detail"] or "invalide" in response.json()["detail"]


def test_load_model_invalid_hf_syntax_rejected_400():
    response = client.post("/api/model/load", json={"model_id": "malicious repo with spaces!"})
    assert response.status_code == 400
    assert "invalide" in response.json()["detail"]


def test_load_model_download_failure_propagates_400(monkeypatch):
    import app.analyzer as analyzer_mod

    def mock_download(*args, **kwargs):
        raise RuntimeError("Hugging Face Hub repository not found: simulated 404")

    monkeypatch.setattr(analyzer_mod, "snapshot_download", mock_download)

    response = client.post("/api/model/load", json={"model_id": "nonexistent-org/nonexistent-model"})
    assert response.status_code == 400
    assert "Impossible de télécharger le modèle HuggingFace" in response.json()["detail"]
    assert "simulated 404" in response.json()["detail"]


def test_cors_headers_allowed_origin():
    # Test that preflight OPTIONS request with allowed origin returns correct headers
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_headers_forbidden_origin():
    # Test that preflight OPTIONS request with forbidden origin does NOT return allow origin
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://unauthorized-evil-site.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") != "http://unauthorized-evil-site.com"


def test_cors_wildcard_logic():
    import os
    from app.main import get_cors_origins

    with patch.dict(os.environ, {"CORS_ORIGINS": "*"}):
        origins = get_cors_origins()
        assert origins == ["*"]
        allow_cred = "*" not in origins
        assert allow_cred is False




