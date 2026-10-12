# 🗺️ Roadmap - MoE Weight & Expert Visualizer

Ce document récapitule la vision du projet et les fonctionnalités implémentées et à venir.

## 🎯 Fonctionnalités réalisées
- [x] **Phase 1 — Moteur d'analyse MoE Backend (FastAPI) & Vue Tableau (React)** :
  - Analyse et extraction des taux d'activation/utilisation des poids et experts d'un modèle Mixture-of-Experts.
  - Domaines d'expertise : Philosophie, Mathématiques, Code, Sciences, Littérature.
  - Vue Tableau interactive avec tri dynamique par pourcentage d'utilisation décroissant.
- [x] **Phase 2 — Topologie du Réseau Neuronal MoE (`MoENetworkGraph`)** :
  - Graphe interactif des 4 couches × 8 experts (32 nœuds d'experts).
  - Liens de co-activation inter-couches pondérés par intensité.
  - Illumination des experts activés et inspection détaillée au clic.
- [x] **Phase 3 — Analyse sur Prompt Personnalisé (`/api/analyze-prompt` & `PromptAnalyzer`)** :
  - Routage dynamique multi-domaines sur saisie libre utilisateur.
  - Affichage des pondérations par domaine détecté et bannière de prévention sur le calibrage des données.
- [x] **Couverture de tests complète (TDD)** : Tests unitaires backend (`pytest`) et frontend (`Vitest`).

---

## 🚀 Prochaines étapes (Phase 4)
- [ ] **Hook Inférence Local Temps Réel** : Connexion directe aux logits du gating network sur `llama.cpp`, `vLLM` ou `Ollama`.
- [ ] **Échelle & Extensibilité** : Support de 50+ domaines d'expertise pré-configurés.
- [ ] **Export des rapports d'analyse** : Export JSON / CSV / PDF.
