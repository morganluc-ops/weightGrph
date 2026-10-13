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
