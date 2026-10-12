from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.analyzer import MoEAnalyzer

app = FastAPI(title="MoE Weight & Expert Analyzer API", version="1.0.0")

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

@app.get("/api/analyze/{domain_id}")
def analyze_domain(domain_id: str):
    try:
        return analyzer.analyze_domain(domain_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/model/info")
def get_model_info():
    return {
        "model_name": "Local MoE Model (Mixtral-8x7B / DeepSeek-MoE equivalent)",
        "architecture": "Mixture of Experts",
        "num_layers": 4,
        "experts_per_layer": 8,
        "total_experts": 32,
        "active_experts_per_tok": 2
    }
