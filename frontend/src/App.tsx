import React, { useState, useEffect } from 'react';
import { DomainSelector } from './components/DomainSelector';
import { ExpertTable } from './components/ExpertTable';
import { MoENetworkGraph } from './components/MoENetworkGraph';
import { PromptAnalyzer } from './components/PromptAnalyzer';
import { ModelManager } from './components/ModelManager';
import {
  fetchDomains,
  fetchDomainAnalysis,
  fetchModelInfo,
  fetchAvailableModels,
  loadMoEModel,
  analyzeCustomPrompt,
} from './api';
import { DomainsMap, DomainAnalysisResult, ModelInfo, MoEModelOption } from './types';

export const App: React.FC = () => {
  const [domains, setDomains] = useState<DomainsMap>({});
  const [models, setModels] = useState<MoEModelOption[]>([]);
  const [selectedDomain, setSelectedDomain] = useState<string>('');
  const [lastCustomPrompt, setLastCustomPrompt] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<DomainAnalysisResult | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingModel, setLoadingModel] = useState<boolean>(false);
  const [loadingTargetId, setLoadingTargetId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function init() {
      try {
        setLoading(true);
        const [domainsData, infoData, modelsData] = await Promise.all([
          fetchDomains(),
          fetchModelInfo(),
          fetchAvailableModels().catch(() => []),
        ]);
        setDomains(domainsData);
        setModelInfo(infoData);
        setModels(modelsData);

        const firstDomain = Object.keys(domainsData)[0];
        if (firstDomain) {
          setSelectedDomain(firstDomain);
          const initialAnalysis = await fetchDomainAnalysis(firstDomain);
          setAnalysis(initialAnalysis);
        }
      } catch (err: any) {
        setError(err.message || 'Erreur d\'initialisation');
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  const handleDomainSelect = async (domainId: string) => {
    setSelectedDomain(domainId);
    setLastCustomPrompt(null);
    try {
      setLoading(true);
      setError(null);
      const data = await fetchDomainAnalysis(domainId);
      setAnalysis(data);
    } catch (err: any) {
      setError(err.message || 'Erreur de chargement');
    } finally {
      setLoading(false);
    }
  };

  const handlePromptAnalyze = async (prompt: string) => {
    setLastCustomPrompt(prompt);
    try {
      setLoading(true);
      setError(null);
      const data = await analyzeCustomPrompt(prompt);
      setAnalysis(data);
    } catch (err: any) {
      setError(err.message || 'Erreur lors de l\'analyse du prompt');
    } finally {
      setLoading(false);
    }
  };

  const handleModelSelect = async (modelId: string) => {
    try {
      setLoadingModel(true);
      setLoadingTargetId(modelId);
      setError(null);

      // Reset analyses to start fresh on the new model
      setAnalysis(null);
      setLastCustomPrompt(null);

      const updatedInfo = await loadMoEModel(modelId);
      setModelInfo(updatedInfo);
      const updatedModels = await fetchAvailableModels().catch(() => models);
      setModels(updatedModels);

      // Recompute initial domain analysis cleanly with the new model
      if (selectedDomain) {
        setLoading(true);
        const data = await fetchDomainAnalysis(selectedDomain);
        setAnalysis(data);
      }
    } catch (err: any) {
      setError(err.message || 'Erreur lors du chargement du modèle MoE');
    } finally {
      setLoadingModel(false);
      setLoadingTargetId(null);
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1040px', margin: '0 auto', padding: '20px', fontFamily: 'system-ui, sans-serif' }}>
      <header style={{ borderBottom: '2px solid #2c3e50', paddingBottom: '15px', marginBottom: '25px' }}>
        <h1 style={{ margin: 0, color: '#2c3e50', fontSize: '28px' }}>
          🧠 MoE Weight & Expert Visualizer
        </h1>
        <p style={{ color: '#7f8c8d', margin: '5px 0 0 0' }}>
          Analyse en temps réel de l'activation des poids et experts d'un modèle Mixture-of-Experts (PyTorch MoE)
        </p>
        {modelInfo && (
          <div style={{ marginTop: '10px', fontSize: '13px', backgroundColor: '#ecf0f1', padding: '8px 12px', borderRadius: '4px' }}>
            <strong>Modèle actif :</strong> {modelInfo.model_name} | <strong>Couches MoE :</strong> {modelInfo.num_layers} | <strong>Experts par couche :</strong> {modelInfo.experts_per_layer} | <strong>Total Experts :</strong> {modelInfo.total_experts} | <strong>Périphérique :</strong> {modelInfo.device?.toUpperCase()}
          </div>
        )}
      </header>

      {error && (
        <div style={{ padding: '12px', backgroundColor: '#e74c3c', color: '#fff', borderRadius: '4px', marginBottom: '15px' }}>
          {error}
        </div>
      )}

      {models.length > 0 && (
        <ModelManager
          models={models}
          currentModel={modelInfo}
          onSelectModel={handleModelSelect}
          loadingModel={loadingModel}
          loadingTargetId={loadingTargetId}
        />
      )}

      {Object.keys(domains).length > 0 && (
        <DomainSelector
          domains={domains}
          selectedDomain={selectedDomain}
          onSelectDomain={handleDomainSelect}
        />
      )}

      <PromptAnalyzer
        onAnalyzePrompt={handlePromptAnalyze}
        loading={loading || loadingModel}
        detectedDomains={analysis?.detected_domains}
        domainsMap={domains}
        tokenRouting={analysis?.token_routing}
        isRealInference={analysis?.is_real_inference}
      />

      {loading && !analysis && (
        <div style={{ padding: '30px', textAlign: 'center', color: '#2563eb', fontWeight: 'bold' }}>
          ⏳ Inférence PyTorch en cours sur le modèle actif...
        </div>
      )}

      {!loading && analysis && (
        <MoENetworkGraph
          analysis={analysis}
          numLayers={modelInfo?.num_layers || 4}
          expertsPerLayer={modelInfo?.experts_per_layer || 8}
        />
      )}

      <ExpertTable analysis={analysis} loading={loading} />
    </div>
  );
};

export default App;
