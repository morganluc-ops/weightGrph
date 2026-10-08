# 🗺️ Roadmap - MoE Weight & Expert Visualizer

Ce document récapitule la vision du projet et les fonctionnalités futures à implémenter.

## 🎯 Fonctionnalités réalisées (v1.0 - Core Engine)
- [x] **Moteur d'analyse MoE Backend (FastAPI)** : Analyse et extraction des taux d'activation/utilisation des poids et experts d'un modèle Mixture-of-Experts.
- [x] **Domaines d'expertise** : Support initial des domaines (Philosophie, Mathématiques, Code, Sciences, Littérature).
- [x] **Vue Tableau interactive (React + TypeScript)** :
  - Sélection de domaines en temps réel.
  - Tri dynamique du plus fort pourcentage au plus faible (% d'utilisation décroissant).
  - Détails des experts (Couche, ID, Nom/Spécialité, Taille des poids en paramètres, Rang).
- [x] **Couverture de tests complète (TDD)** : Tests unitaires backend (`pytest`) et frontend (`Vitest`).

---

## 🚀 Prochaines étapes & Roadmap

### Phase 2 : Rendu 3D Réseau Neuronal (Sphère / Brain Topology)
- [ ] **Graphe 3D Interactif** :
  - Représentation sous forme de sphère 3D / réseau de neurones cérébral avec Three.js / React Three Fiber / `3d-force-graph`.
  - Nœuds représentant les blocs de poids / experts.
  - Liens (arêtes) représentant les flux de co-activation entre couches d'experts.
- [ ] **Illumination & Surbrillance par domaine** :
  - Mise en avant lumineuse (glow) des experts activés pour le domaine sélectionné.
  - Affichage des pourcentages d'utilisation en overlay au-dessus des nœuds 3D.

### Phase 3 : Connexion Inférence Local MoE & Saisie Utilisateur
- [ ] **Hook Inférence Local** :
  - Intégration avec `llama.cpp`, `vLLM`, `Ollama` et PyTorch/Transformers pour capturer en direct les logits du gating network lors d'une inférence sur un modèle MoE local.
- [ ] **Analyse sur Prompt Perso** :
  - Champ de texte permettant à l'utilisateur de passer ses propres textes pour calculer l'activation exacte des poids.
  - Bouton avec message de prévention / alerte sur les risques d'analyse sur de mauvais jeux de données.

### Phase 4 : Échelle & Extensibilité
- [ ] Support d'une cinquantaine (50+) de domaines d'expertise pré-configurés.
- [ ] Export des rapports d'analyse (JSON / CSV / PDF).
