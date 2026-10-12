import pytest
from app.analyzer import MoEAnalyzer, DomainAnalysisResult, ExpertUsage

def test_moe_analyzer_initialization():
    analyzer = MoEAnalyzer()
    domains = analyzer.get_available_domains()
    assert len(domains) > 0
    assert "philosophie" in domains
    assert "mathematiques" in domains

def test_analyze_domain_returns_sorted_experts():
    analyzer = MoEAnalyzer()
    result = analyzer.analyze_domain("philosophie")

    assert isinstance(result, DomainAnalysisResult)
    assert result.domain_id == "philosophie"
    assert len(result.experts) > 0

    # Check sorted descending by usage_percentage
    for i in range(len(result.experts) - 1):
        assert result.experts[i].usage_percentage >= result.experts[i+1].usage_percentage

def test_analyze_unknown_domain_raises_value_error():
    analyzer = MoEAnalyzer()
    with pytest.raises(ValueError, match="Domaine inconnu"):
        analyzer.analyze_domain("domaine_inexistant")

def test_expert_usage_structure():
    expert = ExpertUsage(layer_id=0, expert_id=2, usage_percentage=45.5, num_params="1.5B", expert_name="Expert Philo #2")
    assert expert.layer_id == 0
    assert expert.expert_id == 2
    assert expert.usage_percentage == 45.5

def test_analyze_prompt_detects_domain():
    analyzer = MoEAnalyzer()
    result = analyzer.analyze_prompt("Implémente un algorithme en Python avec FastAPI et résous une équation matricielle.")
    assert result.domain_id == "custom_prompt"
    assert len(result.experts) > 0
    assert result.detected_domains is not None
    assert "code" in result.detected_domains
    assert "mathematiques" in result.detected_domains
    for i in range(len(result.experts) - 1):
        assert result.experts[i].usage_percentage >= result.experts[i + 1].usage_percentage

def test_analyze_empty_prompt_raises_value_error():
    analyzer = MoEAnalyzer()
    with pytest.raises(ValueError, match="Le prompt ne peut pas être vide"):
        analyzer.analyze_prompt("   ")
