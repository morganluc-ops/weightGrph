import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import re
import time
import math
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_REAL_MOE_MODEL = "optimum-intel-internal-testing/tiny-random-qwen1.5-moe"
MAX_PROMPT_LENGTH = 10_000
MAX_ANALYZED_TOKENS = 128
MODELS_DIR = (Path(__file__).resolve().parent.parent / "models").resolve()

HF_MODEL_ID_PATTERN = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}(?:/[a-zA-Z0-9][a-zA-Z0-9._-]{0,99})?$"
)


def is_safe_local_model_path(candidate: str, models_dir: Path) -> Optional[Path]:
    """
    Vérifie si candidate correspond à un dossier de modèle valide et existant sous models_dir.
    Retourne le Path résolu si valide, None sinon.
    """
    clean = candidate.strip()
    if not clean:
        return None

    normalized_parts = clean.replace("\\", "/").split("/")
    if ".." in normalized_parts:
        return None

    try:
        resolved_base = models_dir.resolve()
        p = Path(clean)

        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (resolved_base / p).resolve()

        resolved.relative_to(resolved_base)
        if resolved.is_dir():
            return resolved
    except (ValueError, OSError):
        return None

    return None


def validate_model_id(model_id: str, models_dir: Optional[Path] = None) -> tuple[str, bool]:
    """
    Valide l'identifiant de modèle :
    - Soit un chemin local ou sous-dossier existant confiné sous models_dir (is_local=True)
    - Soit un identifiant HuggingFace valide (is_local=False)
    Retourne (validated_model_id, is_local).
    Lève ValueError si invalide.
    """
    target_dir = (Path(models_dir).resolve() if models_dir else MODELS_DIR)

    if not isinstance(model_id, str):
        raise ValueError("L'identifiant du modèle doit être une chaîne de caractères")

    clean_id = model_id.strip()
    if not clean_id:
        raise ValueError("L'identifiant du modèle ne peut pas être vide")

    if len(clean_id) > 255:
        raise ValueError("L'identifiant du modèle dépasse la taille maximale autorisée (255 caractères)")

    # 1. Vérification contre les path traversal
    normalized_parts = clean_id.replace("\\", "/").split("/")
    if ".." in normalized_parts:
        raise ValueError(f"Identifiant de modèle invalide : tentative de path traversal interdite ('{clean_id}')")

    # 2. Vérifier si c'est un chemin local sous models_dir
    local_path = is_safe_local_model_path(clean_id, target_dir)
    if local_path is not None:
        return str(local_path), True

    p = Path(clean_id)
    if p.is_absolute() or ":" in clean_id or clean_id.startswith("/") or clean_id.startswith("\\"):
        raise ValueError(
            f"Chemin de modèle local non autorisé ou introuvable : '{clean_id}'. "
            "Les modèles locaux doivent impérativement résider sous le répertoire 'models/'."
        )

    # 3. Vérifier le format Hugging Face
    if HF_MODEL_ID_PATTERN.match(clean_id):
        return clean_id, False

    raise ValueError(
        f"Identifiant de modèle invalide : '{clean_id}'. "
        "Doit être un identifiant HuggingFace valide (ex: 'organisation/modele') "
        "ou un dossier local existant dans le répertoire 'models/'."
    )



class ExpertUsage(BaseModel):
    layer_id: int
    expert_id: int
    usage_percentage: float  # Pourcentage de charge globale captée par cet expert par rapport à l'ensemble du modèle (somme des probabilités de routage softmax sur tous les tokens / total des probabilités de tous les experts * 100)
    num_params: str
    expert_name: str


class TopKExpert(BaseModel):
    expert_id: int
    weight: float  # Pourcentage d'activation softmax (ex: 68.4%)


class TokenLayerRouting(BaseModel):
    layer_id: int
    expert_id: int  # Expert dominant (Top-1) sur cette couche
    confidence: float  # Confiance de l'expert dominant (%)
    top_k: List[TopKExpert] = []  # Top-k experts sur cette couche


class TokenRoutingInfo(BaseModel):
    token: str
    layer_id: int = 0
    expert_id: int = 0
    confidence: float = 0.0
    top_k: List[TopKExpert] = []
    layers: Optional[List[TokenLayerRouting]] = None


class LayerBalanceMetrics(BaseModel):
    layer_id: int
    entropy: float  # Entropie de Shannon normalisée entre 0.0 et 1.0 pour cette couche
    gini_coefficient: float  # Indice de Gini discret normalisé [0.0, 1.0] pour cette couche
    dead_experts_count: int  # Nombre d'experts ayant reçu 0 activation dans cette couche
    total_experts: int  # Nombre total d'experts dans cette couche
    max_expert_load_share: float  # Part de charge captée par l'expert le plus actif de la couche (en %)


class BalanceMetrics(BaseModel):
    entropy: float  # Entropie de Shannon normalisée globale [0.0, 1.0] sur tous les experts du modèle
    gini_coefficient: float  # Indice de Gini global [0.0, 1.0] sur l'ensemble des experts du modèle
    dead_experts_count: int  # Nombre total d'experts avec 0 activation sur l'ensemble du modèle
    total_experts: int  # Nombre total d'experts du modèle (num_layers * experts_per_layer)
    max_expert_load_share: float  # Part de charge captée par l'expert le plus sollicité du modèle (en %)
    layers_metrics: List[LayerBalanceMetrics] = []


def compute_balance_metrics(
    expert_scores: Dict[tuple, float],
    num_layers: int,
    experts_per_layer: int,
) -> BalanceMetrics:
    """
    Calcule les métriques d'équilibrage du routage MoE :
    - Entropie normalisée H / ln(N) : 1.0 = distribution parfaitement uniforme, 0.0 = effondrement sur 1 expert.
    - Coefficient de Gini discret normalisé : 0.0 = égalité parfaite, 1.0 = concentration maximale sur 1 expert.
    - Experts morts : nombre d'experts n'ayant reçu aucune activation (charge = 0).
    - Part de charge de l'expert le plus sollicité : pourcentage capté par l'expert le plus actif.
    """
    if num_layers <= 0 or experts_per_layer <= 0:
        return BalanceMetrics(
            entropy=0.0,
            gini_coefficient=0.0,
            dead_experts_count=0,
            total_experts=0,
            max_expert_load_share=0.0,
            layers_metrics=[],
        )

    layers_metrics: List[LayerBalanceMetrics] = []

    # 1. Calcul des métriques par couche
    for l in range(num_layers):
        layer_masses = [expert_scores.get((l, e), 0.0) for e in range(experts_per_layer)]
        tot_layer = sum(layer_masses)
        dead_l = sum(1 for m in layer_masses if m <= 0.0)

        if tot_layer > 0 and experts_per_layer > 1:
            # Entropie normalisée de Shannon pour la couche
            p_layer = [m / tot_layer for m in layer_masses if m > 0.0]
            h_layer = -sum(p * math.log(p) for p in p_layer)
            h_norm_layer = min(1.0, max(0.0, h_layer / math.log(experts_per_layer)))

            # Coefficient de Gini discret normalisé
            sorted_m = sorted(layer_masses)
            gini_num = sum((2 * (i + 1) - experts_per_layer - 1) * sorted_m[i] for i in range(experts_per_layer))
            gini_den = (experts_per_layer - 1) * tot_layer
            gini_layer = min(1.0, max(0.0, gini_num / gini_den)) if gini_den > 0 else 0.0

            max_share_layer = (max(layer_masses) / tot_layer) * 100.0
        elif experts_per_layer == 1:
            h_norm_layer = 1.0
            gini_layer = 0.0
            max_share_layer = 100.0
        else:
            h_norm_layer = 0.0
            gini_layer = 0.0
            max_share_layer = 0.0

        layers_metrics.append(
            LayerBalanceMetrics(
                layer_id=l,
                entropy=round(h_norm_layer, 4),
                gini_coefficient=round(gini_layer, 4),
                dead_experts_count=dead_l,
                total_experts=experts_per_layer,
                max_expert_load_share=round(max_share_layer, 2),
            )
        )

    # 2. Calcul des métriques globales sur tous les experts du modèle
    all_masses = [
        expert_scores.get((l, e), 0.0)
        for l in range(num_layers)
        for e in range(experts_per_layer)
    ]
    total_global_mass = sum(all_masses)
    total_experts = num_layers * experts_per_layer
    total_dead = sum(1 for m in all_masses if m <= 0.0)

    if total_global_mass > 0 and total_experts > 1:
        p_global = [m / total_global_mass for m in all_masses if m > 0.0]
        h_global = -sum(p * math.log(p) for p in p_global)
        h_norm_global = min(1.0, max(0.0, h_global / math.log(total_experts)))

        sorted_all = sorted(all_masses)
        gini_num = sum((2 * (i + 1) - total_experts - 1) * sorted_all[i] for i in range(total_experts))
        gini_den = (total_experts - 1) * total_global_mass
        gini_global = min(1.0, max(0.0, gini_num / gini_den)) if gini_den > 0 else 0.0

        max_share_global = (max(all_masses) / total_global_mass) * 100.0
    elif total_experts == 1:
        h_norm_global = 1.0
        gini_global = 0.0
        max_share_global = 100.0
    else:
        h_norm_global = 0.0
        gini_global = 0.0
        max_share_global = 0.0

    return BalanceMetrics(
        entropy=round(h_norm_global, 4),
        gini_coefficient=round(gini_global, 4),
        dead_experts_count=total_dead,
        total_experts=total_experts,
        max_expert_load_share=round(max_share_global, 2),
        layers_metrics=layers_metrics,
    )


def compute_layer_expert_matrix(
    expert_scores: Dict[tuple, float],
    num_layers: int,
    experts_per_layer: int,
) -> List[List[float]]:
    """
    Calcule la matrice complète couche × expert de l'activation du routage MoE :
    Pour chaque couche (de 0 à num_layers - 1), fournit la liste des activations
    de chaque expert (de 0 à experts_per_layer - 1), normalisée par couche (la ligne somme à 100.0% ou 0.0),
    arrondie à 2 décimales pour optimiser le payload JSON.
    """
    if num_layers <= 0 or experts_per_layer <= 0:
        return []

    matrix: List[List[float]] = []
    for l in range(num_layers):
        layer_masses = [expert_scores.get((l, e), 0.0) for e in range(experts_per_layer)]
        tot_layer = sum(layer_masses)
        if tot_layer > 0:
            row = [round((m / tot_layer) * 100.0, 2) for m in layer_masses]
        else:
            row = [0.0] * experts_per_layer
        matrix.append(row)
    return matrix


class DomainAnalysisResult(BaseModel):
    domain_id: str
    domain_name: str
    experts: List[ExpertUsage]
    detected_domains: Optional[Dict[str, float]] = None
    is_real_inference: bool = True
    model_id: str = DEFAULT_REAL_MOE_MODEL
    token_routing: Optional[List[TokenRoutingInfo]] = None
    truncated: bool = False
    original_token_count: int = 0
    analyzed_token_count: int = 0
    balance_metrics: Optional[BalanceMetrics] = None
    layer_expert_matrix: Optional[List[List[float]]] = None


class PromptAnalysisRequest(BaseModel):
    prompt: str


class LoadModelRequest(BaseModel):
    model_id: str


class DownloadProgressInfo(BaseModel):
    status: str  # "idle" | "downloading" | "loading_tensors" | "ready" | "error"
    model_id: str
    downloaded_bytes: int
    total_bytes: int
    downloaded_mb: float
    total_mb: float
    progress_percent: float
    speed_mb_s: float
    current_file: str
    stage_message: str


AVAILABLE_MOE_MODELS: List[Dict[str, Any]] = [
    {
        "id": "optimum-intel-internal-testing/tiny-random-qwen1.5-moe",
        "name": "Qwen1.5-MoE Compact (PyTorch Safetensors - 4 couches × 8 experts, 40 Mo)",
        "architecture": "Qwen2MoeForCausalLM",
        "num_layers": 4,
        "experts_per_layer": 8,
        "active_experts_per_tok": 2,
        "size_hint": "40 MB (Local)",
        "is_real": True,
    },
    {
        "id": "google/switch-base-8",
        "name": "Google Switch-Base-8 (PyTorch MoE - 6 couches × 8 experts, 1.2 Go)",
        "architecture": "SwitchTransformersForConditionalGeneration",
        "num_layers": 6,
        "experts_per_layer": 8,
        "active_experts_per_tok": 1,
        "size_hint": "1.2 GB",
        "is_real": True,
    },
    {
        "id": "Qwen/Qwen1.5-MoE-A2.7B",
        "name": "Qwen1.5-MoE-A2.7B (PyTorch LLM MoE - 24 couches × 60 experts)",
        "architecture": "Qwen2MoeForCausalLM",
        "num_layers": 24,
        "experts_per_layer": 60,
        "active_experts_per_tok": 4,
        "size_hint": "~5.5 GB",
        "is_real": True,
    },
    {
        "id": "allenai/OLMoE-1B-7B-0924",
        "name": "AllenAI OLMoE-1B-7B (PyTorch LLM MoE - 16 couches × 64 experts)",
        "architecture": "OlmoeForCausalLM",
        "num_layers": 16,
        "experts_per_layer": 64,
        "active_experts_per_tok": 8,
        "size_hint": "~13 GB",
        "is_real": True,
    },
]


class MoEAnalyzer:
    _shared_cache: Dict[str, tuple[Any, Any, Dict[str, Any], str]] = {}
    _lock: threading.RLock = threading.RLock()

    def __init__(self, initial_model_id: str = DEFAULT_REAL_MOE_MODEL, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir).resolve() if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._lock = MoEAnalyzer._lock

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

        self.current_model_id: str = ""
        self.hf_model: Any = None
        self.hf_tokenizer: Any = None
        self.hf_device: str = "cpu"
        self.expert_param_str: str = "1.2M"
        self.model_meta: Dict[str, Any] = {}

        self.progress_state: Dict[str, Any] = {
            "status": "idle",
            "model_id": "",
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "progress_percent": 0.0,
            "speed_mb_s": 0.0,
            "current_file": "",
            "stage_message": "",
        }

        self.load_model(initial_model_id)

    def get_download_progress(self) -> Dict[str, Any]:
        return dict(self.progress_state)

    def get_available_domains(self) -> Dict[str, str]:
        return self.domains_info

    def get_available_models(self) -> List[Dict[str, Any]]:
        with self._lock:
            models = []
            for m in AVAILABLE_MOE_MODELS:
                item = dict(m)
                item["active"] = (m["id"] == self.current_model_id)
                models.append(item)
            if self.current_model_id not in [m["id"] for m in AVAILABLE_MOE_MODELS]:
                models.insert(
                    0,
                    {
                        "id": self.current_model_id,
                        "name": f"{self.current_model_id} (Modèle HuggingFace MoE)",
                        "architecture": self.model_meta.get("architecture", "MoE"),
                        "num_layers": self.model_meta.get("num_layers", 4),
                        "experts_per_layer": self.model_meta.get("experts_per_layer", 8),
                        "active_experts_per_tok": self.model_meta.get("active_experts_per_tok", 2),
                        "size_hint": "HuggingFace Hub",
                        "is_real": True,
                        "active": True,
                    },
                )
            return models

    def get_model_info(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.model_meta)

    def load_model(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            validated_id, is_local = validate_model_id(model_id, self.models_dir)

            if validated_id in MoEAnalyzer._shared_cache:
                model, tokenizer, meta, param_str = MoEAnalyzer._shared_cache[validated_id]
                self.hf_model = model
                self.hf_tokenizer = tokenizer
                self.model_meta = dict(meta)
                self.expert_param_str = param_str
                self.current_model_id = validated_id
                self.hf_device = meta["device"]
                self.progress_state = {
                    "status": "ready",
                    "model_id": validated_id,
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "downloaded_mb": 0.0,
                    "total_mb": 0.0,
                    "progress_percent": 100.0,
                    "speed_mb_s": 0.0,
                    "current_file": "",
                    "stage_message": f"Modèle {validated_id} prêt en mémoire (chargé depuis le cache local)",
                }
                return self.model_meta

            if is_local:
                self.progress_state = {
                    "status": "loading_tensors",
                    "model_id": validated_id,
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "downloaded_mb": 0.0,
                    "total_mb": 0.0,
                    "progress_percent": 100.0,
                    "speed_mb_s": 0.0,
                    "current_file": "",
                    "stage_message": f"Chargement du modèle local {validated_id}...",
                }
            else:
                # Step 1: Pre-download snapshot with live progress tracking
                self.progress_state = {
                    "status": "downloading",
                    "model_id": validated_id,
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "downloaded_mb": 0.0,
                    "total_mb": 0.0,
                    "progress_percent": 0.0,
                    "speed_mb_s": 0.0,
                    "current_file": "Initialisation...",
                    "stage_message": f"Préparation du téléchargement de {validated_id}...",
                }

                progress_ref = self.progress_state
                start_time = time.time()
                last_time = [start_time]
                last_bytes = [0]

                # Calculate estimated total size from metadata
                total_repo_bytes = 0
                try:
                    api = HfApi()
                    info = api.model_info(validated_id, files_metadata=True)
                    for s in info.siblings:
                        fn = s.rfilename.lower()
                        if not any(fn.endswith(ext) for ext in [".msgpack", ".h5", ".ot", ".png", ".jpg"]):
                            total_repo_bytes += (s.size or 0)
                except Exception as e:
                    logger.warning("Impossible de récupérer les métadonnées de taille pour '%s': %s", validated_id, e)
                    total_repo_bytes = 100 * 1024 * 1024  # fallback 100MB

                progress_ref["total_bytes"] = total_repo_bytes
                progress_ref["total_mb"] = round(total_repo_bytes / (1024 * 1024), 1)

                downloaded_bytes_so_far = [0]

                class LiveProgressTqdm(tqdm):
                    def __init__(self, *args, **kwargs):
                        desc = kwargs.get("desc") or ""
                        kwargs["disable"] = True
                        super().__init__(*args, **kwargs)
                        self.file_name = str(desc).split("/")[-1].split("\\")[-1] or "poids"
                        progress_ref["current_file"] = self.file_name
                        progress_ref["stage_message"] = f"Téléchargement : {self.file_name}"

                    def update(self, n=1):
                        super().update(n)
                        downloaded_bytes_so_far[0] += n
                        now = time.time()
                        dt = now - last_time[0]
                        if dt >= 0.25:
                            speed = (downloaded_bytes_so_far[0] - last_bytes[0]) / (dt * 1024 * 1024)
                            progress_ref["speed_mb_s"] = round(max(0.0, speed), 1)
                            last_time[0] = now
                            last_bytes[0] = downloaded_bytes_so_far[0]

                        d_mb = round(downloaded_bytes_so_far[0] / (1024 * 1024), 1)
                        t_mb = progress_ref["total_mb"] or round((self.total or 0) / (1024 * 1024), 1) or 1.0
                        pct = round(min(99.0, (d_mb / t_mb) * 100.0), 1) if t_mb > 0 else 0.0

                        progress_ref["downloaded_bytes"] = downloaded_bytes_so_far[0]
                        progress_ref["downloaded_mb"] = d_mb
                        progress_ref["progress_percent"] = pct
                        progress_ref["stage_message"] = (
                            f"Téléchargement {self.file_name} ({d_mb} Mo / {t_mb} Mo - {pct}%)"
                        )

                try:
                    snapshot_download(
                        repo_id=validated_id,
                        tqdm_class=LiveProgressTqdm,
                        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.png", "*.jpg"],
                    )
                except Exception as e:
                    logger.error(
                        "Échec du téléchargement du modèle HuggingFace '%s': %s",
                        validated_id,
                        e,
                        exc_info=True,
                    )
                    progress_ref["status"] = "error"
                    progress_ref["stage_message"] = f"Échec du téléchargement : {str(e)}"
                    raise ValueError(
                        f"Impossible de télécharger le modèle HuggingFace '{validated_id}': {str(e)}"
                    ) from e

            # Step 2: Model instantiation & tensor loading into PyTorch
            self.progress_state["status"] = "loading_tensors"
            self.progress_state["progress_percent"] = 100.0
            self.progress_state["stage_message"] = "Montage des tenseurs et allocation VRAM/RAM..."

            try:
                config = AutoConfig.from_pretrained(validated_id, trust_remote_code=False)
                tokenizer = AutoTokenizer.from_pretrained(validated_id, trust_remote_code=False)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32

                if getattr(config, "is_encoder_decoder", False) or "switch" in getattr(config, "model_type", "").lower():
                    model = AutoModelForSeq2SeqLM.from_pretrained(validated_id, torch_dtype=dtype, trust_remote_code=False)
                else:
                    model = AutoModelForCausalLM.from_pretrained(validated_id, torch_dtype=dtype, trust_remote_code=False)

                model.to(device)
                model.eval()

                num_experts = (
                    getattr(config, "num_experts", None)
                    or getattr(config, "num_local_experts", None)
                    or getattr(config, "n_routed_experts", None)
                    or 8
                )
                num_layers_total = getattr(config, "num_layers", None) or getattr(config, "num_hidden_layers", 4)
                sparse_step = getattr(config, "encoder_sparse_step", 1) or 1
                if "switch" in getattr(config, "model_type", "").lower() and sparse_step > 1:
                    num_moe_layers = (num_layers_total // sparse_step) * 2
                else:
                    num_moe_layers = num_layers_total

                active_k = (
                    getattr(config, "num_experts_per_tok", None)
                    or getattr(config, "num_selected_experts", None)
                    or 2
                )

                total_params = sum(p.numel() for p in model.parameters())
                expert_params_approx = total_params / max(1, num_moe_layers * num_experts)
                if expert_params_approx >= 1e9:
                    param_str = f"{expert_params_approx / 1e9:.2f}B"
                elif expert_params_approx >= 1e6:
                    param_str = f"{expert_params_approx / 1e6:.2f}M"
                else:
                    param_str = f"{expert_params_approx / 1e3:.1f}K"

                arch_list = getattr(config, "architectures", None)
                arch_name = arch_list[0] if arch_list else getattr(config, "model_type", "Mixture of Experts")

                meta = {
                    "model_name": f"{validated_id} ({arch_name} sur {device.upper()})",
                    "model_id": validated_id,
                    "architecture": arch_name,
                    "num_layers": int(num_moe_layers),
                    "experts_per_layer": int(num_experts),
                    "total_experts": int(num_moe_layers * num_experts),
                    "active_experts_per_tok": int(active_k),
                    "is_real_model": True,
                    "device": device,
                }

                MoEAnalyzer._shared_cache[validated_id] = (model, tokenizer, meta, param_str)
                self.hf_model = model
                self.hf_tokenizer = tokenizer
                self.hf_device = device
                self.current_model_id = validated_id
                self.expert_param_str = param_str
                self.model_meta = dict(meta)

                self.progress_state["status"] = "ready"
                self.progress_state["stage_message"] = f"Modèle {validated_id} monté et prêt !"
                return self.model_meta
            except Exception as e:
                logger.error(
                    "Échec du chargement en mémoire du modèle '%s': %s",
                    validated_id,
                    e,
                    exc_info=True,
                )
                self.progress_state["status"] = "error"
                self.progress_state["stage_message"] = f"Erreur : {str(e)}"
                raise ValueError(f"Impossible de charger le modèle MoE '{validated_id}': {str(e)}") from e

    def _run_real_moe_inference(
        self, text: str
    ) -> tuple[List[ExpertUsage], List[TokenRoutingInfo], BalanceMetrics, List[List[float]], bool, int, int]:
        with self._lock:
            # Calcul du nombre total de tokens avant troncature pour signaler l'écrêtage éventuel
            full_encoded = self.hf_tokenizer.encode(text, add_special_tokens=True)
            original_token_count = len(full_encoded)

            inputs = self.hf_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_ANALYZED_TOKENS,
            ).to(self.hf_device)
            input_ids = inputs["input_ids"]
            tokens = self.hf_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            analyzed_token_count = len(tokens)
            truncated = original_token_count > analyzed_token_count

            with torch.no_grad():
                if getattr(self.hf_model.config, "is_encoder_decoder", False):
                    # Pour les modèles Encoder-Decoder (ex: Google Switch), l'utilisation d'input_ids dans decoder_input_ids
                    # introduisait un biais artificiel de teacher-forcing. Nous fournissons ici le token de démarrage standard
                    # du décodeur pour respecter la dynamique d'inférence réelle.
                    dec_start_id = getattr(self.hf_model.config, "decoder_start_token_id", None)
                    if dec_start_id is None:
                        dec_start_id = getattr(self.hf_model.config, "pad_token_id", 0) or 0
                    decoder_input_ids = torch.full(
                        (input_ids.shape[0], 1),
                        dec_start_id,
                        dtype=torch.long,
                        device=self.hf_device,
                    )
                    outputs = self.hf_model(
                        input_ids=input_ids,
                        decoder_input_ids=decoder_input_ids,
                        output_router_logits=True,
                        return_dict=True,
                    )
                    raw_layers = []
                    # Les couches MoE de l'encodeur analysent directement les tokens du prompt
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
            active_k = int(self.model_meta.get("active_experts_per_tok", 2))
            self.model_meta["num_layers"] = num_moe_layers
            self.model_meta["experts_per_layer"] = num_experts
            self.model_meta["total_experts"] = num_moe_layers * num_experts

            expert_scores: Dict[tuple, float] = {}
            layer_topk_indices = []
            layer_topk_vals = []
            k = min(active_k, num_experts)

            for layer_idx, logits in enumerate(moe_layer_tensors):
                logits_2d = logits[0] if logits.ndim == 3 else logits
                probs = torch.softmax(logits_2d, dim=-1)
                seq_len = probs.shape[0]

                topk_vals, topk_indices = torch.topk(probs, k=k, dim=-1)
                layer_topk_vals.append(topk_vals)
                layer_topk_indices.append(topk_indices)

                for tok_idx in range(seq_len):
                    for rank_i in range(k):
                        exp_id = int(topk_indices[tok_idx, rank_i].item())
                        gate_prob = float(topk_vals[tok_idx, rank_i].item())
                        key = (layer_idx, exp_id)
                        expert_scores[key] = expert_scores.get(key, 0.0) + gate_prob

            token_routing: List[TokenRoutingInfo] = []

            for tok_idx in range(len(tokens)):
                raw_tok = tokens[tok_idx]
                if isinstance(raw_tok, bytes):
                    raw_tok = raw_tok.decode("utf-8", errors="replace")
                clean_tok = str(raw_tok).replace("▁", " ").replace("Ġ", " ").strip() or str(raw_tok)

                layers_routing: List[TokenLayerRouting] = []
                for layer_idx in range(num_moe_layers):
                    topk_i = layer_topk_indices[layer_idx]
                    topk_v = layer_topk_vals[layer_idx]
                    seq_len_l = topk_i.shape[0]

                    if tok_idx < seq_len_l:
                        layer_top_k: List[TopKExpert] = []
                        for rank_i in range(k):
                            exp_id = int(topk_i[tok_idx, rank_i].item())
                            weight_val = round(float(topk_v[tok_idx, rank_i].item()) * 100.0, 1)
                            layer_top_k.append(TopKExpert(expert_id=exp_id, weight=weight_val))

                        dom_exp = layer_top_k[0].expert_id if layer_top_k else 0
                        dom_conf = layer_top_k[0].weight if layer_top_k else 0.0
                        layers_routing.append(
                            TokenLayerRouting(
                                layer_id=layer_idx,
                                expert_id=dom_exp,
                                confidence=dom_conf,
                                top_k=layer_top_k,
                            )
                        )
                    else:
                        layers_routing.append(
                            TokenLayerRouting(
                                layer_id=layer_idx,
                                expert_id=0,
                                confidence=0.0,
                                top_k=[],
                            )
                        )

                l0_routing = layers_routing[0] if layers_routing else None
                l0_exp = l0_routing.expert_id if l0_routing else 0
                l0_conf = l0_routing.confidence if l0_routing else 0.0
                l0_top_k = l0_routing.top_k if l0_routing else []

                token_routing.append(
                    TokenRoutingInfo(
                        token=clean_tok,
                        layer_id=0,
                        expert_id=l0_exp,
                        confidence=l0_conf,
                        top_k=l0_top_k,
                        layers=layers_routing,
                    )
                )

            total_mass = sum(expert_scores.values()) or 1.0
            experts_list: List[ExpertUsage] = []
            for (layer_id, expert_id), mass in expert_scores.items():
                pct = round((mass / total_mass) * 100.0, 2)
                if pct > 0.0:
                    experts_list.append(
                        ExpertUsage(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            usage_percentage=pct,
                            num_params=self.expert_param_str,
                            expert_name=f"MoE Layer {layer_id} - Expert {expert_id} (Gating Top-{active_k} Réel)",
                        )
                    )

            experts_list.sort(key=lambda x: x.usage_percentage, reverse=True)
            balance_metrics = compute_balance_metrics(expert_scores, num_moe_layers, num_experts)
            layer_expert_matrix = compute_layer_expert_matrix(expert_scores, num_moe_layers, num_experts)
            return (
                experts_list[:16],
                token_routing,
                balance_metrics,
                layer_expert_matrix,
                truncated,
                original_token_count,
                analyzed_token_count,
            )

    def analyze_domain(self, domain_id: str) -> DomainAnalysisResult:
        with self._lock:
            if domain_id not in self.domains_info:
                raise ValueError(f"Domaine inconnu: {domain_id}")

            sample_text = self.domain_corpus[domain_id]
            (
                experts_models,
                token_routing,
                balance_metrics,
                layer_expert_matrix,
                truncated,
                orig_count,
                analyzed_count,
            ) = self._run_real_moe_inference(sample_text)
            return DomainAnalysisResult(
                domain_id=domain_id,
                domain_name=self.domains_info[domain_id],
                experts=experts_models,
                is_real_inference=True,
                model_id=self.current_model_id,
                token_routing=token_routing,
                truncated=truncated,
                original_token_count=orig_count,
                analyzed_token_count=analyzed_count,
                balance_metrics=balance_metrics,
                layer_expert_matrix=layer_expert_matrix,
            )

    def analyze_prompt(self, prompt: str) -> DomainAnalysisResult:
        with self._lock:
            if not prompt or not prompt.strip():
                raise ValueError("Le prompt ne peut pas être vide")

            if len(prompt) > MAX_PROMPT_LENGTH:
                raise ValueError(
                    f"Le prompt dépasse la limite maximale autorisée de {MAX_PROMPT_LENGTH} caractères"
                )

            (
                experts_models,
                token_routing,
                balance_metrics,
                layer_expert_matrix,
                truncated,
                orig_count,
                analyzed_count,
            ) = self._run_real_moe_inference(prompt.strip())
            short_prompt = prompt.strip()[:60] + ("..." if len(prompt.strip()) > 60 else "")
            return DomainAnalysisResult(
                domain_id="custom_prompt",
                domain_name=f"Prompt réel : « {short_prompt} »",
                experts=experts_models,
                is_real_inference=True,
                model_id=self.current_model_id,
                token_routing=token_routing,
                truncated=truncated,
                original_token_count=orig_count,
                analyzed_token_count=analyzed_count,
                balance_metrics=balance_metrics,
                layer_expert_matrix=layer_expert_matrix,
            )
