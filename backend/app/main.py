from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.analyzer import MoEAnalyzer, PromptAnalysisRequest, LoadModelRequest

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
        import os
        prompts_dir = "prompts"
        # Check if the prompt is an id of a prompt file
        if payload.prompt.endswith(".md"):
            prompt_file = os.path.join(prompts_dir, payload.prompt)
            if os.path.exists(prompt_file):
                with open(prompt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                return analyzer.analyze_prompt(content)

        return analyzer.analyze_prompt(payload.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/prompts")
def get_prompts():
    import os
    prompts_dir = "prompts"
    if not os.path.exists(prompts_dir):
        return []

    files = [f for f in os.listdir(prompts_dir) if f.endswith(".md")]
    return [{"id": f, "name": f.replace(".md", "")} for f in files]

@app.get("/api/model/info")
def get_model_info():
    return analyzer.get_model_info()
