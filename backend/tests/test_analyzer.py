import pytest
from app.analyzer import (
    MoEAnalyzer,
    DomainAnalysisResult,
    ExpertUsage,
    DEFAULT_REAL_MOE_MODEL,
    BalanceMetrics,
    LayerBalanceMetrics,
    compute_balance_metrics,
)

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


def test_compute_balance_metrics_uniform_distribution():
    # 2 couches x 4 experts, tous activés avec la même masse de 1.0
    scores = {(l, e): 1.0 for l in range(2) for e in range(4)}
    metrics = compute_balance_metrics(scores, num_layers=2, experts_per_layer=4)

    assert metrics.total_experts == 8
    assert metrics.dead_experts_count == 0
    assert metrics.entropy == 1.0
    assert metrics.gini_coefficient == 0.0
    assert metrics.max_expert_load_share == round((1.0 / 8.0) * 100.0, 2)
    assert len(metrics.layers_metrics) == 2

    for lm in metrics.layers_metrics:
        assert lm.total_experts == 4
        assert lm.dead_experts_count == 0
        assert lm.entropy == 1.0
        assert lm.gini_coefficient == 0.0
        assert lm.max_expert_load_share == 25.0


def test_compute_balance_metrics_complete_collapse():
    # 2 couches x 4 experts, un seul expert activé à 100%
    scores = {(0, 0): 10.0}
    metrics = compute_balance_metrics(scores, num_layers=2, experts_per_layer=4)

    assert metrics.total_experts == 8
    assert metrics.dead_experts_count == 7
    assert metrics.entropy == 0.0
    assert metrics.gini_coefficient == 1.0
    assert metrics.max_expert_load_share == 100.0

    # Layer 0: 1 active, 3 dead
    lm0 = metrics.layers_metrics[0]
    assert lm0.dead_experts_count == 3
    assert lm0.entropy == 0.0
    assert lm0.gini_coefficient == 1.0
    assert lm0.max_expert_load_share == 100.0

    # Layer 1: 0 active, 4 dead
    lm1 = metrics.layers_metrics[1]
    assert lm1.dead_experts_count == 4
    assert lm1.entropy == 0.0
    assert lm1.gini_coefficient == 0.0
    assert lm1.max_expert_load_share == 0.0


def test_compute_balance_metrics_partial_dead_experts():
    # 1 couche x 8 experts: 4 experts à 1.0, 4 experts morts à 0.0
    scores = {(0, e): 1.0 for e in range(4)}
    metrics = compute_balance_metrics(scores, num_layers=1, experts_per_layer=8)

    assert metrics.total_experts == 8
    assert metrics.dead_experts_count == 4
    # log(4) / log(8) = 2/3 ≈ 0.6667
    assert abs(metrics.entropy - 0.6667) < 0.001
    assert 0.0 < metrics.gini_coefficient < 1.0
    assert metrics.max_expert_load_share == 25.0


def test_compute_balance_metrics_edge_cases():
    # 0 couches / 0 experts
    empty_res = compute_balance_metrics({}, num_layers=0, experts_per_layer=0)
    assert empty_res.total_experts == 0
    assert empty_res.dead_experts_count == 0
    assert empty_res.entropy == 0.0
    assert empty_res.gini_coefficient == 0.0

    # Toutes les masses à 0
    zero_res = compute_balance_metrics({}, num_layers=2, experts_per_layer=4)
    assert zero_res.total_experts == 8
    assert zero_res.dead_experts_count == 8
    assert zero_res.entropy == 0.0
    assert zero_res.gini_coefficient == 0.0


def test_analyze_domain_and_prompt_contain_balance_metrics():
    analyzer = MoEAnalyzer()
    domain_res = analyzer.analyze_domain("philosophie")
    assert domain_res.balance_metrics is not None
    bm = domain_res.balance_metrics
    assert isinstance(bm, BalanceMetrics)
    assert 0.0 <= bm.entropy <= 1.0
    assert 0.0 <= bm.gini_coefficient <= 1.0
    assert 0 <= bm.dead_experts_count <= bm.total_experts
    assert 0.0 < bm.max_expert_load_share <= 100.0
    assert len(bm.layers_metrics) == 4

    prompt_res = analyzer.analyze_prompt("Résoudre les équations différentielles en physique quantique")
    assert prompt_res.balance_metrics is not None
    pbm = prompt_res.balance_metrics
    assert 0.0 <= pbm.entropy <= 1.0
    assert 0.0 <= pbm.gini_coefficient <= 1.0
    assert 0 <= pbm.dead_experts_count <= pbm.total_experts
    assert 0.0 < pbm.max_expert_load_share <= 100.0

