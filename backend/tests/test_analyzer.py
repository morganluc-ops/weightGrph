import pytest
from app.analyzer import MoEAnalyzer, DomainAnalysisResult, ExpertUsage, DEFAULT_REAL_MOE_MODEL

def test_moe_analyzer_initialization():
    analyzer = MoEAnalyzer()
    domains = analyzer.get_available_domains()
    assert len(domains) > 0
    assert "philosophie" in domains
    assert "mathematiques" in domains
    assert analyzer.get_model_info()["is_real_model"] is True

def test_analyze_domain_returns_sorted_real_experts():
    analyzer = MoEAnalyzer()
    result = analyzer.analyze_domain("philosophie")

    assert isinstance(result, DomainAnalysisResult)
    assert result.domain_id == "philosophie"
    assert result.is_real_inference is True
    assert len(result.experts) > 0

    for i in range(len(result.experts) - 1):
        assert result.experts[i].usage_percentage >= result.experts[i + 1].usage_percentage

def test_analyze_unknown_domain_raises_value_error():
    analyzer = MoEAnalyzer()
    with pytest.raises(ValueError, match="Domaine inconnu"):
        analyzer.analyze_domain("domaine_inexistant")

def test_expert_usage_structure():
    expert = ExpertUsage(layer_id=0, expert_id=2, usage_percentage=45.5, num_params="1.2M", expert_name="Expert #2")
    assert expert.layer_id == 0
    assert expert.expert_id == 2
    assert expert.usage_percentage == 45.5

def test_analyze_prompt_real_router_logits():
    analyzer = MoEAnalyzer()
    result = analyzer.analyze_prompt("Implémente un algorithme en Python avec FastAPI et résous une équation matricielle.")
    assert result.domain_id == "custom_prompt"
    assert result.is_real_inference is True
    assert len(result.experts) > 0
    assert result.token_routing is not None and len(result.token_routing) > 0
    for i in range(len(result.experts) - 1):
        assert result.experts[i].usage_percentage >= result.experts[i + 1].usage_percentage

def test_different_prompts_produce_distinct_real_routing():
    analyzer = MoEAnalyzer()
    r1 = analyzer.analyze_prompt("Écris une fonction Python récursive pour trier un tableau")
    r2 = analyzer.analyze_prompt("La conscience morale chez Kant et Platon")
    sig1 = [(e.layer_id, e.expert_id, e.usage_percentage) for e in r1.experts]
    sig2 = [(e.layer_id, e.expert_id, e.usage_percentage) for e in r2.experts]
    assert sig1 != sig2

def test_analyze_empty_prompt_raises_value_error():
    analyzer = MoEAnalyzer()
    with pytest.raises(ValueError, match="Le prompt ne peut pas être vide"):
        analyzer.analyze_prompt("   ")

def test_available_models_catalog():
    analyzer = MoEAnalyzer()
    models = analyzer.get_available_models()
    ids = [m["id"] for m in models]
    assert DEFAULT_REAL_MOE_MODEL in ids
    assert "google/switch-base-8" in ids
    assert all(m["is_real"] is True for m in models)
