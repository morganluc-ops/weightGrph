from pydantic import BaseModel
from typing import List, Dict

class ExpertUsage(BaseModel):
    layer_id: int
    expert_id: int
    usage_percentage: float
    num_params: str
    expert_name: str

class DomainAnalysisResult(BaseModel):
    domain_id: str
    domain_name: str
    experts: List[ExpertUsage]

class MoEAnalyzer:
    def __init__(self):
        self.domains_info: Dict[str, str] = {
            "philosophie": "Philosophie & Éthique",
            "mathematiques": "Mathématiques & Logique",
            "code": "Programmation & Génie Logiciel",
            "sciences": "Sciences Physiques & Naturelles",
            "litterature": "Littérature & Langues"
        }

        self.expert_usage_db: Dict[str, List[Dict]] = {
            "philosophie": [
                {"layer_id": 0, "expert_id": 3, "usage_percentage": 37.4, "num_params": "1.5B", "expert_name": "MoE Layer 0 - Expert 3 (Conceptuel)"},
                {"layer_id": 1, "expert_id": 1, "usage_percentage": 28.2, "num_params": "1.5B", "expert_name": "MoE Layer 1 - Expert 1 (Éthique)"},
                {"layer_id": 2, "expert_id": 7, "usage_percentage": 14.8, "num_params": "1.5B", "expert_name": "MoE Layer 2 - Expert 7 (Sémantique)"},
                {"layer_id": 0, "expert_id": 2, "usage_percentage": 9.5, "num_params": "1.5B", "expert_name": "MoE Layer 0 - Expert 2 (Logique)"},
                {"layer_id": 3, "expert_id": 5, "usage_percentage": 6.1, "num_params": "1.5B", "expert_name": "MoE Layer 3 - Expert 5 (Synthèse)"},
                {"layer_id": 1, "expert_id": 0, "usage_percentage": 4.0, "num_params": "1.5B", "expert_name": "MoE Layer 1 - Expert 0 (Général)"},
            ],
            "mathematiques": [
                {"layer_id": 0, "expert_id": 2, "usage_percentage": 42.1, "num_params": "1.5B", "expert_name": "MoE Layer 0 - Expert 2 (Algèbre)"},
                {"layer_id": 2, "expert_id": 4, "usage_percentage": 31.0, "num_params": "1.5B", "expert_name": "MoE Layer 2 - Expert 4 (Calcul)"},
                {"layer_id": 1, "expert_id": 6, "usage_percentage": 15.3, "num_params": "1.5B", "expert_name": "MoE Layer 1 - Expert 6 (Géométrie)"},
                {"layer_id": 3, "expert_id": 1, "usage_percentage": 8.4, "num_params": "1.5B", "expert_name": "MoE Layer 3 - Expert 1 (Formel)"},
                {"layer_id": 0, "expert_id": 0, "usage_percentage": 3.2, "num_params": "1.5B", "expert_name": "MoE Layer 0 - Expert 0 (Base)"},
            ],
            "code": [
                {"layer_id": 1, "expert_id": 5, "usage_percentage": 45.0, "num_params": "1.5B", "expert_name": "MoE Layer 1 - Expert 5 (Syntaxe Code)"},
                {"layer_id": 3, "expert_id": 2, "usage_percentage": 25.5, "num_params": "1.5B", "expert_name": "MoE Layer 3 - Expert 2 (Algorithmes)"},
                {"layer_id": 0, "expert_id": 1, "usage_percentage": 18.0, "num_params": "1.5B", "expert_name": "MoE Layer 0 - Expert 1 (Type Checker)"},
                {"layer_id": 2, "expert_id": 3, "usage_percentage": 11.5, "num_params": "1.5B", "expert_name": "MoE Layer 2 - Expert 3 (Optimization)"},
            ],
            "sciences": [
                {"layer_id": 2, "expert_id": 1, "usage_percentage": 39.8, "num_params": "1.5B", "expert_name": "MoE Layer 2 - Expert 1 (Physique)"},
                {"layer_id": 0, "expert_id": 4, "usage_percentage": 30.2, "num_params": "1.5B", "expert_name": "MoE Layer 0 - Expert 4 (Chimie)"},
                {"layer_id": 3, "expert_id": 0, "usage_percentage": 20.0, "num_params": "1.5B", "expert_name": "MoE Layer 3 - Expert 0 (Biologie)"},
                {"layer_id": 1, "expert_id": 2, "usage_percentage": 10.0, "num_params": "1.5B", "expert_name": "MoE Layer 1 - Expert 2 (Data)"},
            ],
            "litterature": [
                {"layer_id": 0, "expert_id": 7, "usage_percentage": 41.2, "num_params": "1.5B", "expert_name": "MoE Layer 0 - Expert 7 (Stylistique)"},
                {"layer_id": 1, "expert_id": 3, "usage_percentage": 33.8, "num_params": "1.5B", "expert_name": "MoE Layer 1 - Expert 3 (Grammaire & Vocabulaire)"},
                {"layer_id": 3, "expert_id": 6, "usage_percentage": 16.0, "num_params": "1.5B", "expert_name": "MoE Layer 3 - Expert 6 (Narration)"},
                {"layer_id": 2, "expert_id": 0, "usage_percentage": 9.0, "num_params": "1.5B", "expert_name": "MoE Layer 2 - Expert 0 (Sens Contextuel)"},
            ]
        }

    def get_available_domains(self) -> Dict[str, str]:
        return self.domains_info

    def analyze_domain(self, domain_id: str) -> DomainAnalysisResult:
        if domain_id not in self.expert_usage_db:
            raise ValueError(f"Domaine inconnu: {domain_id}")

        raw_experts = self.expert_usage_db[domain_id]
        sorted_experts = sorted(raw_experts, key=lambda x: x["usage_percentage"], reverse=True)

        experts_models = [
            ExpertUsage(
                layer_id=item["layer_id"],
                expert_id=item["expert_id"],
                usage_percentage=item["usage_percentage"],
                num_params=item["num_params"],
                expert_name=item["expert_name"]
            )
            for item in sorted_experts
        ]

        return DomainAnalysisResult(
            domain_id=domain_id,
            domain_name=self.domains_info[domain_id],
            experts=experts_models
        )
