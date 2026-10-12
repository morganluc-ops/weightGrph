from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import hashlib
import math
import re

class ExpertUsage(BaseModel):
    layer_id: int
    expert_id: int
    usage_percentage: float
    num_params: str
    expert_name: str

class TokenRoutingInfo(BaseModel):
    token: str
    layer_id: int
    expert_id: int
    confidence: float

class DomainAnalysisResult(BaseModel):
    domain_id: str
    domain_name: str
    experts: List[ExpertUsage]
    detected_domains: Optional[Dict[str, float]] = None
    is_real_inference: bool = False
    model_id: str = "simulated"
    token_routing: Optional[List[TokenRoutingInfo]] = None

class PromptAnalysisRequest(BaseModel):
    prompt: str

class LoadModelRequest(BaseModel):
    model_id: str

AVAILABLE_MOE_MODELS: List[Dict[str, Any]] = [
    {
        "id": "google/switch-base-8",
        "name": "Google Switch-Base-8 (Vrai MoE PyTorch - 8 experts/couche, ~490 Mo)",
        "architecture": "SwitchTransformers MoE",
        "num_layers": 6,
        "experts_per_layer": 8,
        "active_experts_per_tok": 1,
        "size_hint": "~490 MB",
        "is_real": True,
    },
    {
        "id": "google/switch-base-16",
        "name": "Google Switch-Base-16 (Vrai MoE PyTorch - 16 experts/couche, ~950 Mo)",
        "architecture": "SwitchTransformers MoE",
        "num_layers": 6,
        "experts_per_layer": 16,
        "active_experts_per_tok": 1,
        "size_hint": "~950 MB",
        "is_real": True,
    },
    {
        "id": "allenai/OLMoE-1B-7B-0924",
        "name": "AllenAI OLMoE-1B-7B (Vrai LLM MoE - 64 experts/couche, ~13 Go FP16)",
        "architecture": "OLMoE CausalLM",
        "num_layers": 16,
        "experts_per_layer": 64,
        "active_experts_per_tok": 8,
        "size_hint": "~13 GB",
        "is_real": True,
    },
    {
        "id": "simulated",
        "name": "Local MoE Model (Mixtral-8x7B / DeepSeek-MoE equivalent)",
        "architecture": "Mixture of Experts",
        "num_layers": 4,
        "experts_per_layer": 8,
        "active_experts_per_tok": 2,
        "size_hint": "0 MB (Intégré)",
        "is_real": False,
    },
]


class MoEAnalyzer:
    def __init__(self):
        self.current_model_id: str = "simulated"
        self.hf_model: Any = None
        self.hf_tokenizer: Any = None
        self.hf_device: str = "cpu"
        self.model_meta: Dict[str, Any] = {
            "model_name": "Local MoE Model (Mixtral-8x7B / DeepSeek-MoE equivalent)",
            "model_id": "simulated",
            "architecture": "Mixture of Experts",
            "num_layers": 4,
            "experts_per_layer": 8,
            "total_experts": 32,
            "active_experts_per_tok": 2,
            "is_real_model": False,
            "device": "cpu",
        }

        self.domains_info: Dict[str, str] = {
            "philosophie": "Philosophie & Éthique",
            "mathematiques": "Mathématiques & Logique",
            "code": "Programmation & Génie Logiciel",
            "sciences": "Sciences Physiques & Naturelles",
            "litterature": "Littérature & Langues",
        }

        self.domain_corpus: Dict[str, str] = {
            "philosophie": (
                "La conscience morale et le libre arbitre constituent le fondement de l'éthique chez Kant. "
                "Socrate et Platon interrogent la vérité, la justice et l'ontologie de l'être humain."
            ),
            "mathematiques": (
                "Soit E un espace vectoriel normé. Calculons l'intégrale convergente, le déterminant de la matrice "
                "et démontrons le théorème d'algèbre linéaire par récurrence."
            ),
            "code": (
                "def analyze_router_logits(tensor: list[float]) -> dict:\n"
                "    return {i: round(v, 4) for i, v in enumerate(sorted(tensor, reverse=True))}\n"
                "FastAPI async endpoint TypeScript React component."
            ),
            "sciences": (
                "En physique quantique et thermodynamique, l'intrication des particules subatomiques "
                "et la réplication de l'ADN cellulaire en biologie moléculaire gouvernent le vivant."
            ),
            "litterature": (
                "Les métaphores poétiques de Baudelaire et la narration romanesque de Proust "
                "illustrent la richesse stylistique et grammaticale de la littérature classique."
            ),
        }

        self.domain_keywords: Dict[str, List[str]] = {
            "philosophie": [
                "philosoph", "éthique", "ethique", "morale", "conscience", "liberté", "liberte",
                "kant", "socrate", "platon", "aristote", "nietzsche", "descartes", "être", "etre",
                "existence", "justice", "vérité", "verite", "métaphysique", "ontologie", "stoïcisme", "âme", "esprit"
            ],
            "mathematiques": [
                "math", "équation", "equation", "algèbre", "algebre", "calcul", "intégrale", "integrale",
                "dérivée", "derivee", "matrice", "vecteur", "théorème", "theoreme", "probabilité",
                "géométrie", "geometrie", "nombre", "fonction", "topologie", "statistique", "logique", "somme", "suite"
            ],
            "code": [
                "code", "python", "typescript", "javascript", "react", "api", "fastapi", "fonction",
                "algorithme", "variable", "classe", "git", "bug", "compile", "serveur", "sql", "docker",
                "programme", "développ", "developp", "boucle", "array", "json", "http", "rust", "c++", "script", "html"
            ],
            "sciences": [
                "science", "physique", "chimie", "biologie", "atome", "molécule", "molecule", "énergie",
                "energie", "cellule", "adn", "génétique", "genetique", "quantique", "relativité",
                "gravité", "gravite", "univers", "neurone", "cerveau", "climat", "thermodynamique", "planète"
            ],
            "litterature": [
                "littérature", "litterature", "roman", "poésie", "poesie", "poème", "poeme", "écrivain",
                "ecrivain", "auteur", "récit", "recit", "métaphore", "metaphore", "style", "grammaire",
                "langue", "texte", "narrateur", "personnage", "hugo", "baudelaire", "proust", "fable", "livre"
            ]
        }

        # Complete catalog of 32 experts (4 layers x 8 experts) for granular token-level routing
        self.expert_specialties: Dict[tuple, str] = {
            (0, 0): "Lexique Général & Tokenisation",
            (0, 1): "Typage & Structures Syntaxiques",
            (0, 2): "Algèbre & Symboles Formels",
            (0, 3): "Concepts Abstraits & Ontologie",
            (0, 4): "Nomenclature Chimique & Moléculaire",
            (0, 5): "Morphologie & Racines Verbales",
            (0, 6): "Opérateurs Arithmétiques",
            (0, 7): "Figures de Style & Rhétorique",
            (1, 0): "Contexte Discursif Général",
            (1, 1): "Raisonnement Éthique & Déontologie",
            (1, 2): "Analyse Empirique & Données",
            (1, 3): "Grammaire & Syntaxe Littéraire",
            (1, 4): "Cinématique & Lois Physiques",
            (1, 5): "AST & Grammaire de Programmation",
            (1, 6): "Géométrie & Espaces Vectoriels",
            (1, 7): "Argumentation & Dialectique",
            (2, 0): "Sémantique Contextuelle",
            (2, 1): "Mécanique Quantique & Relativité",
            (2, 2): "Preuves & Induction Logique",
            (2, 3): "Optimisation & Complexité Algorithmique",
            (2, 4): "Calcul Différentiel & Intégral",
            (2, 5): "Biologie Cellulaire & Génétique",
            (2, 6): "Prosodie & Rythme Poétique",
            (2, 7): "Herméneutique & Sémantique",
            (3, 0): "Systèmes Biologiques & Écologie",
            (3, 1): "Formalisme & Vérification",
            (3, 2): "Architecture Logicielle & Design Patterns",
            (3, 3): "Épistémologie & Critique",
            (3, 4): "Modélisation Statistique",
            (3, 5): "Synthèse Conceptuelle",
            (3, 6): "Structure Narrative & Récit",
            (3, 7): "Cohérence Globale & Décodage",
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
            ],
        }

    def get_available_domains(self) -> Dict[str, str]:
        return self.domains_info

    def get_available_models(self) -> List[Dict[str, Any]]:
        models = []
        for m in AVAILABLE_MOE_MODELS:
            item = dict(m)
            item["active"] = (m["id"] == self.current_model_id)
            models.append(item)
        if self.current_model_id not in [m["id"] for m in AVAILABLE_MOE_MODELS]:
            models.insert(0, {
                "id": self.current_model_id,
                "name": f"{self.current_model_id} (Modèle HuggingFace personnalisé)",
                "architecture": self.model_meta["architecture"],
                "num_layers": self.model_meta["num_layers"],
                "experts_per_layer": self.model_meta["experts_per_layer"],
                "active_experts_per_tok": self.model_meta["active_experts_per_tok"],
                "size_hint": "HuggingFace Hub",
                "is_real": True,
                "active": True,
            })
        return models

    def get_model_info(self) -> Dict[str, Any]:
        return self.model_meta

    def load_model(self, model_id: str) -> Dict[str, Any]:
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("L'identifiant du modèle ne peut pas être vide")

        if model_id == "simulated":
            self.current_model_id = "simulated"
            self.hf_model = None
            self.hf_tokenizer = None
            self.model_meta = {
                "model_name": "Local MoE Model (Mixtral-8x7B / DeepSeek-MoE equivalent)",
                "model_id": "simulated",
                "architecture": "Mixture of Experts",
                "num_layers": 4,
                "experts_per_layer": 8,
                "total_experts": 32,
                "active_experts_per_tok": 2,
                "is_real_model": False,
                "device": "cpu",
            }
            return self.model_meta

        try:
            import torch
            from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
        except ImportError as e:
            raise ValueError(f"PyTorch et Transformers sont requis pour charger un vrai modèle MoE : {e}")

        try:
            config = AutoConfig.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            if getattr(config, "is_encoder_decoder", False) or "switch" in getattr(config, "model_type", "").lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

            model.to(device)
            model.eval()

            # Inspect MoE dimensions from config
            num_experts = (
                getattr(config, "num_experts", None)
                or getattr(config, "num_local_experts", None)
                or getattr(config, "n_routed_experts", None)
                or 8
            )
            num_layers_total = getattr(config, "num_layers", None) or getattr(config, "num_hidden_layers", 6)
            sparse_step = getattr(config, "encoder_sparse_step", 1) or 1
            if "switch" in getattr(config, "model_type", "").lower() and sparse_step > 1:
                num_moe_layers = (num_layers_total // sparse_step) * 2  # encoder + decoder MoE layers
            else:
                num_moe_layers = num_layers_total

            active_k = (
                getattr(config, "num_experts_per_tok", None)
                or getattr(config, "num_selected_experts", None)
                or 1
            )

            total_params = sum(p.numel() for p in model.parameters())
            expert_params_approx = total_params / max(1, num_moe_layers * num_experts)
            if expert_params_approx >= 1e9:
                param_str = f"{expert_params_approx / 1e9:.1f}B"
            else:
                param_str = f"{expert_params_approx / 1e6:.1f}M"

            self.hf_model = model
            self.hf_tokenizer = tokenizer
            self.hf_device = device
            self.current_model_id = model_id
            self.expert_param_str = param_str

            self.model_meta = {
                "model_name": f"{model_id} (Inférence PyTorch Réelle sur {device.upper()})",
                "model_id": model_id,
                "architecture": getattr(config, "architectures", [config.model_type])[0],
                "num_layers": int(num_moe_layers),
                "experts_per_layer": int(num_experts),
                "total_experts": int(num_moe_layers * num_experts),
                "active_experts_per_tok": int(active_k),
                "is_real_model": True,
                "device": device,
            }
            return self.model_meta
        except Exception as e:
            raise ValueError(f"Impossible de charger le modèle MoE '{model_id}': {str(e)}")

    def _run_real_moe_inference(self, text: str) -> tuple[List[ExpertUsage], List[TokenRoutingInfo]]:
        import torch

        inputs = self.hf_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.hf_device)
        input_ids = inputs["input_ids"]
        tokens = self.hf_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        with torch.no_grad():
            if getattr(self.hf_model.config, "is_encoder_decoder", False):
                outputs = self.hf_model(
                    input_ids=input_ids,
                    decoder_input_ids=input_ids,
                    output_router_logits=True,
                    return_dict=True,
                )
                raw_layers = []
                if getattr(outputs, "encoder_router_logits", None):
                    raw_layers.extend(outputs.encoder_router_logits)
                if getattr(outputs, "decoder_router_logits", None):
                    raw_layers.extend(outputs.decoder_router_logits)
            else:
                outputs = self.hf_model(
                    **inputs,
                    output_router_logits=True,
                    return_dict=True,
                )
                raw_layers = list(getattr(outputs, "router_logits", []) or [])

        # Extract valid router logit tensors per MoE layer
        moe_layer_tensors = []
        for entry in raw_layers:
            if entry is None:
                continue
            tensor = None
            if isinstance(entry, torch.Tensor):
                tensor = entry
            elif isinstance(entry, (tuple, list)):
                for sub in entry:
                    if isinstance(sub, torch.Tensor) and sub.ndim >= 2 and sub.shape[-1] > 1:
                        tensor = sub
                        break
            if tensor is not None and tensor.numel() > 0 and tensor.shape[-1] > 1:
                moe_layer_tensors.append(tensor.detach().float().cpu())

        num_moe_layers = len(moe_layer_tensors)
        if num_moe_layers == 0:
            raise ValueError("Ce modèle n'expose aucun tenseur router_logits MoE.")

        num_experts = int(moe_layer_tensors[0].shape[-1])
        # Update metadata dynamically if layer count differs
        self.model_meta["num_layers"] = num_moe_layers
        self.model_meta["experts_per_layer"] = num_experts
        self.model_meta["total_experts"] = num_moe_layers * num_experts

        expert_scores: Dict[tuple, float] = {}
        token_routing: List[TokenRoutingInfo] = []

        for layer_idx, logits in enumerate(moe_layer_tensors):
            # Shape can be (batch, seq_len, num_experts) or (seq_len, num_experts)
            if logits.ndim == 3:
                logits_2d = logits[0]
            else:
                logits_2d = logits

            probs = torch.softmax(logits_2d, dim=-1)
            seq_len = probs.shape[0]

            for tok_idx in range(seq_len):
                tok_probs = probs[tok_idx]
                top_prob, top_expert = torch.max(tok_probs, dim=-1)
                exp_id = int(top_expert.item())
                prob_val = float(top_prob.item())

                # Accumulate router gating mass across all experts in this layer
                for e_id in range(num_experts):
                    key = (layer_idx, e_id)
                    expert_scores[key] = expert_scores.get(key, 0.0) + float(tok_probs[e_id].item())

                if layer_idx == 0 and tok_idx < len(tokens):
                    clean_tok = tokens[tok_idx].replace("▁", " ").replace("Ġ", " ").strip() or tokens[tok_idx]
                    token_routing.append(
                        TokenRoutingInfo(
                            token=clean_tok,
                            layer_id=layer_idx,
                            expert_id=exp_id,
                            confidence=round(prob_val * 100.0, 1),
                        )
                    )

        total_mass = sum(expert_scores.values()) or 1.0
        param_str = getattr(self, "expert_param_str", "15.2M")

        experts_list: List[ExpertUsage] = []
        for (layer_id, expert_id), mass in expert_scores.items():
            pct = round((mass / total_mass) * 100.0, 2)
            if pct > 0.0:
                stage = "Encodeur" if layer_id < (num_moe_layers // 2) else "Décodeur"
                experts_list.append(
                    ExpertUsage(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        usage_percentage=pct,
                        num_params=param_str,
                        expert_name=f"{self.current_model_id} — Couche {layer_id} ({stage}) Expert {expert_id}",
                    )
                )

        experts_list.sort(key=lambda x: x.usage_percentage, reverse=True)
        return experts_list[:12], token_routing[:24]

    def analyze_domain(self, domain_id: str) -> DomainAnalysisResult:
        if domain_id not in self.expert_usage_db:
            raise ValueError(f"Domaine inconnu: {domain_id}")

        if self.current_model_id != "simulated" and self.hf_model is not None:
            sample_text = self.domain_corpus.get(domain_id, domain_id)
            experts_models, token_routing = self._run_real_moe_inference(sample_text)
            return DomainAnalysisResult(
                domain_id=domain_id,
                domain_name=f"{self.domains_info[domain_id]} (Mesure réelle sur {self.current_model_id})",
                experts=experts_models,
                is_real_inference=True,
                model_id=self.current_model_id,
                token_routing=token_routing,
            )

        raw_experts = self.expert_usage_db[domain_id]
        sorted_experts = sorted(raw_experts, key=lambda x: x["usage_percentage"], reverse=True)

        experts_models = [
            ExpertUsage(
                layer_id=item["layer_id"],
                expert_id=item["expert_id"],
                usage_percentage=item["usage_percentage"],
                num_params=item["num_params"],
                expert_name=item["expert_name"],
            )
            for item in sorted_experts
        ]

        return DomainAnalysisResult(
            domain_id=domain_id,
            domain_name=self.domains_info[domain_id],
            experts=experts_models,
            is_real_inference=False,
            model_id="simulated",
        )

    def _DetectDomains(self, prompt: str) -> Dict[str, float]:
        normalized = prompt.lower().strip()
        words = re.findall(r"[\wàâäéèêëîïôöùûüç\+]+", normalized)

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
            digest = hashlib.sha256(normalized.encode("utf-8")).digest()
            domain_keys = list(self.domains_info.keys())
            primary = domain_keys[digest[0] % len(domain_keys)]
            secondary = domain_keys[digest[1] % len(domain_keys)]
            domain_scores[primary] = 6.5 + (digest[2] % 30) / 10.0
            domain_scores[secondary] = domain_scores.get(secondary, 0.0) + 2.5 + (digest[3] % 20) / 10.0
            total_score = sum(domain_scores.values())

        return {
            d: round((score / total_score) * 100.0, 1)
            for d, score in domain_scores.items()
            if score > 0
        }

    def analyze_prompt(self, prompt: str) -> DomainAnalysisResult:
        if not prompt or not prompt.strip():
            raise ValueError("Le prompt ne peut pas être vide")

        domain_weights = self._DetectDomains(prompt)
        dominant_domain = max(domain_weights.items(), key=lambda x: x[1])[0]
        dominant_label = self.domains_info[dominant_domain]

        # Real PyTorch MoE inference when a real model is loaded
        if self.current_model_id != "simulated" and self.hf_model is not None:
            experts_models, token_routing = self._run_real_moe_inference(prompt)
            return DomainAnalysisResult(
                domain_id="custom_prompt",
                domain_name=f"Prompt analysé par {self.current_model_id} (Dominante : {dominant_label})",
                experts=experts_models,
                detected_domains=domain_weights,
                is_real_inference=True,
                model_id=self.current_model_id,
                token_routing=token_routing,
            )

        # Token-level dynamic gating simulation when in simulated mode
        # Combines domain priors with per-token trigonometric gating logits so every prompt yields unique activations
        words = re.findall(r"[\wàâäéèêëîïôöùûüç\+\-]+", prompt.strip())
        if not words:
            words = [prompt.strip()]

        expert_mass: Dict[tuple, float] = {}
        token_routing: List[TokenRoutingInfo] = []

        # Seed prior from matched domains
        for domain_id, weight_pct in domain_weights.items():
            weight_factor = weight_pct / 100.0
            for exp in self.expert_usage_db[domain_id]:
                key = (exp["layer_id"], exp["expert_id"])
                expert_mass[key] = expert_mass.get(key, 0.0) + exp["usage_percentage"] * weight_factor * 0.55

        # Per-token gating pass across all 4 layers x 8 experts
        for tok_idx, tok in enumerate(words[:32]):
            tok_hash = hashlib.md5(f"{tok.lower()}_{tok_idx}".encode("utf-8")).digest()
            for layer_id in range(4):
                layer_logits = []
                for exp_id in range(8):
                    prior = expert_mass.get((layer_id, exp_id), 1.0)
                    byte_val = tok_hash[(layer_id * 2 + exp_id) % 16]
                    logit = math.log(max(0.5, prior)) + (byte_val / 255.0) * 2.2
                    layer_logits.append(logit)

                max_l = max(layer_logits)
                exps = [math.exp(l - max_l) for l in layer_logits]
                sum_e = sum(exps)
                probs = [e / sum_e for e in exps]

                # Top-2 gating per token
                ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
                top1_id, top1_p = ranked[0]
                top2_id, top2_p = ranked[1]

                expert_mass[(layer_id, top1_id)] = expert_mass.get((layer_id, top1_id), 0.0) + top1_p * 18.0
                expert_mass[(layer_id, top2_id)] = expert_mass.get((layer_id, top2_id), 0.0) + top2_p * 9.0

                if layer_id == 0:
                    token_routing.append(
                        TokenRoutingInfo(
                            token=tok,
                            layer_id=0,
                            expert_id=top1_id,
                            confidence=round(top1_p * 100.0, 1),
                        )
                    )

        total_mass = sum(expert_mass.values()) or 1.0
        merged_list = []
        for (l_id, e_id), mass in expert_mass.items():
            pct = round((mass / total_mass) * 100.0, 1)
            if pct >= 1.5:
                spec = self.expert_specialties.get((l_id, e_id), "Routage Général")
                merged_list.append(
                    ExpertUsage(
                        layer_id=l_id,
                        expert_id=e_id,
                        usage_percentage=pct,
                        num_params="1.5B",
                        expert_name=f"MoE Layer {l_id} - Expert {e_id} ({spec})",
                    )
                )

        merged_list.sort(key=lambda x: x.usage_percentage, reverse=True)
        top_experts = merged_list[:8]

        return DomainAnalysisResult(
            domain_id="custom_prompt",
            domain_name=f"Prompt personnalisé (Dominante : {dominant_label})",
            experts=top_experts,
            detected_domains=domain_weights,
            is_real_inference=False,
            model_id="simulated",
            token_routing=token_routing[:20],
        )
