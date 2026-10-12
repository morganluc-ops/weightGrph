import pytest
from fastapi.testclient import TestClient
from app.main import app

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
    assert "experts" in data
    assert len(data["experts"]) > 0

    # Check descending order
    experts = data["experts"]
    for i in range(len(experts) - 1):
        assert experts[i]["usage_percentage"] >= experts[i+1]["usage_percentage"]

def test_analyze_invalid_domain():
    response = client.get("/api/analyze/invalid_domain")
    assert response.status_code == 404
    assert response.json()["detail"] == "Domaine inconnu: invalid_domain"

def test_get_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    info = response.json()
    assert info["model_name"] == "Local MoE Model (Mixtral-8x7B / DeepSeek-MoE equivalent)"
    assert info["num_layers"] == 4
    assert info["experts_per_layer"] == 8
