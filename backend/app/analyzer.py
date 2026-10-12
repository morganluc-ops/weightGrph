from pydantic import BaseModel
from typing import List, Dict, Optional
import hashlib
import re

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
    detected_domains: Optional[Dict[str, float]] = None

class PromptAnalysisRequest(BaseModel):
    prompt: str

class MoEAnalyzer:
    def __init__(self):
        self.domains_info: Dict[str, str] = {
            "philosophie": "Philosophie & Éthique",
            "mathematiques": "Mathématiques & Logique",
            "code": "Programmation & Génie Logiciel",
            "sciences": "Sciences Physiques & Naturelles",
            "litterature": "Littérature & Langues"
        }

        self.domain_keywords: Dict[str, List[str]] = {
            "philosophie": [
                "philosoph", "éthique", "ethique", "morale", "conscience", "liberté", "liberte",
                "kant", "socrate", "platon", "aristote", "nietzsche", "descartes", "être", "etre",
                "existence", "justice", "vérité", "verite", "métaphysique", "ontologie", "stoïcisme"
            ],
            "mathematiques": [
                "math", "équation", "equation", "algèbre", "algebre", "calcul", "intégrale", "integrale",
                "dérivée", "derivee", "matrice", "vecteur", "théorème", "theoreme", "probabilité",
                "géométrie", "geometrie", "nombre", "fonction", "topologie", "statistique", "logique"
            ],
            "code": [
                "code", "python", "typescript", "javascript", "react", "api", "fastapi", "fonction",
                "algorithme", "variable", "classe", "git", "bug", "compile", "serveur", "sql", "docker",
                "programme", "développ", "developp", "boucle", "array", "json", "http", "rust", "c++"
            ],
            "sciences": [
                "science", "physique", "chimie", "biologie", "atome", "molécule", "molecule", "énergie",
                "energie", "cellule", "adn", "génétique", "genetique", "quantique", "relativité",
                "gravité", "gravite", "univers", "neurone", "cerveau", "climat", "thermodynamique"
            ],
            "litterature": [
                "littérature", "litterature", "roman", "poésie", "poesie", "poème", "poeme", "écrivain",
                "ecrivain", "auteur", "récit", "recit", "métaphore", "metaphore", "style", "grammaire",
                "langue", "texte", "narrateur", "personnage", "UGO", "hugo", "baudelaire", "proust", "fable"
            ]
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

    def analyze_prompt(self, prompt: str) -> DomainAnalysisResult:
        if not prompt or not prompt.strip():
            raise ValueError("Le prompt ne peut pas être vide")

        normalized = prompt.lower().strip()
        words = re.findall(r"[\wàâäéèêëîïôöùûüç\+]+", normalized)

        # Score each domain based on keyword matches
        domain_scores: Dict[str, float] = {d: 0.0 for d in self.domains_info}
        for domain_id, keywords in self.domain_keywords.items():
            for kw in keywords:
                if kw in normalized:
                    domain_scores[domain_id] += 2.5
                for w in words:
                    if w.startswith(kw[:4]) and len(kw) >= 4:
                        domain_scores[domain_id] += 1.0

        total_score = sum(domain_scores.values())
        if total_score == 0:
            # Deterministic fallback routing based on prompt hash
            digest = hashlib.sha256(normalized.encode("utf-8")).digest()
            domain_keys = list(self.domains_info.keys())
            primary = domain_keys[digest[0] % len(domain_keys)]
            secondary = domain_keys[digest[1] % len(domain_keys)]
            domain_scores[primary] = 7.0
            domain_scores[secondary] = domain_scores.get(secondary, 0.0) + 3.0
            total_score = sum(domain_scores.values())

        domain_weights = {
            d: round((score / total_score) * 100.0, 1)
            for d, score in domain_scores.items()
            if score > 0
        }

        # Aggregate expert activations across matched domains
        merged_experts: Dict[tuple, Dict] = {}
        for domain_id, weight_pct in domain_weights.items():
            weight_factor = weight_pct / 100.0
            for exp in self.expert_usage_db[domain_id]:
                key = (exp["layer_id"], exp["expert_id"])
                contribution = exp["usage_percentage"] * weight_factor
                if key not in merged_experts:
                    merged_experts[key] = {
                        "layer_id": exp["layer_id"],
                        "expert_id": exp["expert_id"],
                        "usage_percentage": contribution,
                        "num_params": exp["num_params"],
                        "expert_name": exp["expert_name"],
                    }
                else:
                    merged_experts[key]["usage_percentage"] += contribution

        # Normalize top experts to 100%
        expert_list = list(merged_experts.values())
        total_usage = sum(e["usage_percentage"] for e in expert_list) or 1.0
        for e in expert_list:
            e["usage_percentage"] = round((e["usage_percentage"] / total_usage) * 100.0, 1)

        sorted_experts = sorted(expert_list, key=lambda x: x["usage_percentage"], reverse=True)[:8]

        dominant_domain = max(domain_weights.items(), key=lambda x: x[1])[0]
        dominant_label = self.domains_info[dominant_domain]

        return DomainAnalysisResult(
            domain_id="custom_prompt",
            domain_name=f"Prompt personnalisé (Dominante : {dominant_label})",
            experts=[ExpertUsage(**item) for item in sorted_experts],
            detected_domains=domain_weights
        )
