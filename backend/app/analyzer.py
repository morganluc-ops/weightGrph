import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import time
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from huggingface_hub import HfApi, snapshot_download

DEFAULT_REAL_MOE_MODEL = "optimum-intel-internal-testing/tiny-random-qwen1.5-moe"


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
    is_real_inference: bool = True
    model_id: str = DEFAULT_REAL_MOE_MODEL
    token_routing: Optional[List[TokenRoutingInfo]] = None


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

    def __init__(self, initial_model_id: str = DEFAULT_REAL_MOE_MODEL):
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
        return self.progress_state

    def get_available_domains(self) -> Dict[str, str]:
        return self.domains_info

    def get_available_models(self) -> List[Dict[str, Any]]:
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
        return self.model_meta

    def load_model(self, model_id: str) -> Dict[str, Any]:
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("L'identifiant du modèle ne peut pas être vide")

        if model_id in MoEAnalyzer._shared_cache:
            model, tokenizer, meta, param_str = MoEAnalyzer._shared_cache[model_id]
            self.hf_model = model
            self.hf_tokenizer = tokenizer
            self.model_meta = dict(meta)
            self.expert_param_str = param_str
            self.current_model_id = model_id
            self.hf_device = meta["device"]
            self.progress_state = {
                "status": "ready",
                "model_id": model_id,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "downloaded_mb": 0.0,
                "total_mb": 0.0,
                "progress_percent": 100.0,
                "speed_mb_s": 0.0,
                "current_file": "",
                "stage_message": f"Modèle {model_id} prêt en mémoire (chargé depuis le cache local)",
            }
            return self.model_meta

        # Step 1: Pre-download snapshot with live progress tracking
        self.progress_state = {
            "status": "downloading",
            "model_id": model_id,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "progress_percent": 0.0,
            "speed_mb_s": 0.0,
            "current_file": "Initialisation...",
            "stage_message": f"Préparation du téléchargement de {model_id}...",
        }

        progress_ref = self.progress_state
        start_time = time.time()
        last_time = [start_time]
        last_bytes = [0]

        # Calculate estimated total size from metadata
        total_repo_bytes = 0
        try:
            api = HfApi()
            info = api.model_info(model_id, files_metadata=True)
            for s in info.siblings:
                fn = s.rfilename.lower()
                if not any(fn.endswith(ext) for ext in [".msgpack", ".h5", ".ot", ".png", ".jpg"]):
                    total_repo_bytes += (s.size or 0)
        except Exception:
            total_repo_bytes = 100 * 1024 * 1024  # fallback 100MB

        progress_ref["total_bytes"] = total_repo_bytes
        progress_ref["total_mb"] = round(total_repo_bytes / (1024 * 1024), 1)

        downloaded_bytes_so_far = [0]

        class LiveProgressTqdm:
            def __init__(self, *args, total=None, desc=None, **kwargs):
                self.total = total or 0
                self.desc = desc or ""
                self.file_name = self.desc.split("/")[-1].split("\\")[-1] or "poids"
                self.n = 0
                progress_ref["current_file"] = self.file_name
                progress_ref["stage_message"] = f"Téléchargement : {self.file_name}"

            def update(self, n=1):
                self.n += n
                downloaded_bytes_so_far[0] += n
                now = time.time()
                dt = now - last_time[0]
                if dt >= 0.25:
                    speed = (downloaded_bytes_so_far[0] - last_bytes[0]) / (dt * 1024 * 1024)
                    progress_ref["speed_mb_s"] = round(max(0.0, speed), 1)
                    last_time[0] = now
                    last_bytes[0] = downloaded_bytes_so_far[0]

                d_mb = round(downloaded_bytes_so_far[0] / (1024 * 1024), 1)
                t_mb = progress_ref["total_mb"] or round(self.total / (1024 * 1024), 1) or 1.0
                pct = round(min(99.0, (d_mb / t_mb) * 100.0), 1) if t_mb > 0 else 0.0

                progress_ref["downloaded_bytes"] = downloaded_bytes_so_far[0]
                progress_ref["downloaded_mb"] = d_mb
                progress_ref["progress_percent"] = pct
                progress_ref["stage_message"] = (
                    f"Téléchargement {self.file_name} ({d_mb} Mo / {t_mb} Mo - {pct}%)"
                )

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        try:
            snapshot_download(
                repo_id=model_id,
                tqdm_class=LiveProgressTqdm,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.png", "*.jpg"],
            )
        except Exception as e:
            # If snapshot download fails, let from_pretrained report exact error
            pass

        # Step 2: Model instantiation & tensor loading into PyTorch
        progress_ref["status"] = "loading_tensors"
        progress_ref["progress_percent"] = 100.0
        progress_ref["stage_message"] = "Montage des tenseurs et allocation VRAM/RAM..."

        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            if getattr(config, "is_encoder_decoder", False) or "switch" in getattr(config, "model_type", "").lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=False)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=False)

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
                "model_name": f"{model_id} ({arch_name} sur {device.upper()})",
                "model_id": model_id,
                "architecture": arch_name,
                "num_layers": int(num_moe_layers),
                "experts_per_layer": int(num_experts),
                "total_experts": int(num_moe_layers * num_experts),
                "active_experts_per_tok": int(active_k),
                "is_real_model": True,
                "device": device,
            }

            MoEAnalyzer._shared_cache[model_id] = (model, tokenizer, meta, param_str)
            self.hf_model = model
            self.hf_tokenizer = tokenizer
            self.hf_device = device
            self.current_model_id = model_id
            self.expert_param_str = param_str
            self.model_meta = dict(meta)

            progress_ref["status"] = "ready"
            progress_ref["stage_message"] = f"Modèle {model_id} monté et prêt !"
            return self.model_meta
        except Exception as e:
            progress_ref["status"] = "error"
            progress_ref["stage_message"] = f"Erreur : {str(e)}"
            raise ValueError(f"Impossible de charger le modèle MoE '{model_id}': {str(e)}")

    def _run_real_moe_inference(self, text: str) -> tuple[List[ExpertUsage], List[TokenRoutingInfo]]:
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
        token_routing: List[TokenRoutingInfo] = []

        for layer_idx, logits in enumerate(moe_layer_tensors):
            logits_2d = logits[0] if logits.ndim == 3 else logits
            probs = torch.softmax(logits_2d, dim=-1)
            seq_len = probs.shape[0]

            k = min(active_k, num_experts)
            topk_vals, topk_indices = torch.topk(probs, k=k, dim=-1)

            for tok_idx in range(seq_len):
                for rank_i in range(k):
                    exp_id = int(topk_indices[tok_idx, rank_i].item())
                    gate_prob = float(topk_vals[tok_idx, rank_i].item())
                    key = (layer_idx, exp_id)
                    expert_scores[key] = expert_scores.get(key, 0.0) + gate_prob

                if layer_idx == 0 and tok_idx < len(tokens):
                    top1_exp = int(topk_indices[tok_idx, 0].item())
                    top1_prob = float(topk_vals[tok_idx, 0].item())
                    raw_tok = tokens[tok_idx]
                    if isinstance(raw_tok, bytes):
                        raw_tok = raw_tok.decode("utf-8", errors="replace")
                    clean_tok = str(raw_tok).replace("▁", " ").replace("Ġ", " ").strip() or str(raw_tok)
                    token_routing.append(
                        TokenRoutingInfo(
                            token=clean_tok,
                            layer_id=0,
                            expert_id=top1_exp,
                            confidence=round(top1_prob * 100.0, 1),
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
        return experts_list[:16], token_routing[:32]

    def analyze_domain(self, domain_id: str) -> DomainAnalysisResult:
        if domain_id not in self.domains_info:
            raise ValueError(f"Domaine inconnu: {domain_id}")

        sample_text = self.domain_corpus[domain_id]
        experts_models, token_routing = self._run_real_moe_inference(sample_text)
        return DomainAnalysisResult(
            domain_id=domain_id,
            domain_name=self.domains_info[domain_id],
            experts=experts_models,
            is_real_inference=True,
            model_id=self.current_model_id,
            token_routing=token_routing,
        )

    def analyze_prompt(self, prompt: str) -> DomainAnalysisResult:
        if not prompt or not prompt.strip():
            raise ValueError("Le prompt ne peut pas être vide")

        experts_models, token_routing = self._run_real_moe_inference(prompt.strip())
        short_prompt = prompt.strip()[:60] + ("..." if len(prompt.strip()) > 60 else "")
        return DomainAnalysisResult(
            domain_id="custom_prompt",
            domain_name=f"Prompt réel : « {short_prompt} »",
            experts=experts_models,
            is_real_inference=True,
            model_id=self.current_model_id,
            token_routing=token_routing,
        )
