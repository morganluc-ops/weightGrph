import { DomainsMap, DomainAnalysisResult, ModelInfo } from './types';

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
