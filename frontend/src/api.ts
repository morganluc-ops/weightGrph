import { DomainsMap, DomainAnalysisResult, ModelInfo, MoEModelOption, DownloadProgressInfo } from './types';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

export async function fetchDomains(): Promise<DomainsMap> {
  const response = await fetch(`${API_BASE_URL}/domains`);
  if (!response.ok) {
    throw new Error('Erreur lors du chargement des domaines');
  }
  return response.json();
}

export async function fetchDomainAnalysis(domainId: string): Promise<DomainAnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/analyze/${domainId}`);
  if (!response.ok) {
    throw new Error(`Erreur lors de l'analyse du domaine: ${domainId}`);
  }
  const data: DomainAnalysisResult = await response.json();
  data.experts.sort((a, b) => b.usage_percentage - a.usage_percentage);
  return data;
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/model/info`);
  if (!response.ok) {
    throw new Error('Erreur lors du chargement des informations du modèle');
  }
  return response.json();
}

export async function fetchAvailableModels(): Promise<MoEModelOption[]> {
  const response = await fetch(`${API_BASE_URL}/models`);
  if (!response.ok) {
    throw new Error('Erreur lors du chargement de la liste des modèles MoE');
  }
  return response.json();
}

export async function fetchModelProgress(): Promise<DownloadProgressInfo> {
  const response = await fetch(`${API_BASE_URL}/model/progress`);
  if (!response.ok) {
    throw new Error('Erreur lors de la récupération de la progression');
  }
  return response.json();
}

export async function loadMoEModel(modelId: string): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/model/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Erreur lors du chargement du modèle ${modelId}`);
  }
  return response.json();
}

export async function analyzeCustomPrompt(prompt: string): Promise<DomainAnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/analyze-prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  if (!response.ok) {
    throw new Error('Erreur lors de l\'analyse du prompt personnalisé');
  }
  const data: DomainAnalysisResult = await response.json();
  data.experts.sort((a, b) => b.usage_percentage - a.usage_percentage);
  return data;
}
