import React, { useState, useEffect } from 'react';
import { DomainSelector } from './components/DomainSelector';
import { ExpertTable } from './components/ExpertTable';
import { fetchDomains, fetchDomainAnalysis, fetchModelInfo } from './api';
import { DomainsMap, DomainAnalysisResult, ModelInfo } from './types';

export const App: React.FC = () => {
  const [domains, setDomains] = useState<DomainsMap>({});
  const [selectedDomain, setSelectedDomain] = useState<string>('');
  const [analysis, setAnalysis] = useState<DomainAnalysisResult | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function init() {
      try {
        setLoading(true);
        const [domainsData, infoData] = await Promise.all([
          fetchDomains(),
          fetchModelInfo(),
        ]);
        setDomains(domainsData);
        setModelInfo(infoData);

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

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '20px', fontFamily: 'system-ui, sans-serif' }}>
      <header style={{ borderBottom: '2px solid #2c3e50', paddingBottom: '15px', marginBottom: '25px' }}>
        <h1 style={{ margin: 0, color: '#2c3e50', fontSize: '28px' }}>
          🧠 MoE Weight & Expert Visualizer
        </h1>
        <p style={{ color: '#7f8c8d', margin: '5px 0 0 0' }}>
          Analyse de l'utilisation des poids et experts d'un modèle Mixture-of-Experts par domaine
        </p>
        {modelInfo && (
          <div style={{ marginTop: '10px', fontSize: '13px', backgroundColor: '#ecf0f1', padding: '8px 12px', borderRadius: '4px' }}>
            <strong>Modèle détecté :</strong> {modelInfo.model_name} | <strong>Couches :</strong> {modelInfo.num_layers} | <strong>Experts par couche :</strong> {modelInfo.experts_per_layer} | <strong>Total Experts :</strong> {modelInfo.total_experts}
          </div>
        )}
      </header>

      {error && (
        <div style={{ padding: '12px', backgroundColor: '#e74c3c', color: '#fff', borderRadius: '4px', marginBottom: '15px' }}>
          {error}
        </div>
      )}

      {Object.keys(domains).length > 0 && (
        <DomainSelector
          domains={domains}
          selectedDomain={selectedDomain}
          onSelectDomain={handleDomainSelect}
        />
      )}

      <ExpertTable analysis={analysis} loading={loading} />
    </div>
  );
};

export default App;
