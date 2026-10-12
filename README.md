# 🧠 weightGrph — MoE Weight & Expert Visualizer

Application d'analyse et de visualisation de l'activation des poids et experts d'un modèle **Mixture-of-Experts (MoE)** (type Mixtral-8x7B / DeepSeek-MoE, 4 couches × 8 experts) par domaine d'expertise ou sur prompt libre.

## ✨ Fonctionnalités

- **Analyse par Domaine (`/api/analyze/{domain_id}`)** : Philosophie, Mathématiques, Programmation & Code, Sciences, Littérature.
- **Analyse sur Prompt Personnalisé (`POST /api/analyze-prompt`)** : Détection multi-domaines et agrégation pondérée des activations Top-K des experts.
- **Visualisation Topologique du Réseau MoE (`MoENetworkGraph`)** : Graphe interactif des 32 experts répartis sur 4 couches avec flux de co-activation inter-couches et inspecteur d'expert.
- **Vue Tabulaire Classée (`ExpertTable`)** : Classement décroissant des experts par taux d'utilisation (`%`) et taille des poids.

## 🛠️ Architecture

- **Backend** : Python, FastAPI, Pydantic, Pytest (`backend/`)
- **Frontend** : React 18, TypeScript, Vite, Vitest, Testing Library (`frontend/`)

## 🚀 Installation & Démarrage

### 1. Backend (FastAPI — port `8000`)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend (React + Vite — port `5173`)

```powershell
cd frontend
npm install
npm test
npm run dev
```