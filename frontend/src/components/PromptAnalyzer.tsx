import React, { useState } from 'react';
import { DomainsMap } from '../types';

interface PromptAnalyzerProps {
  onAnalyzePrompt: (prompt: string) => void;
  loading: boolean;
  detectedDomains?: { [domainId: string]: number };
  domainsMap: DomainsMap;
}

const SAMPLE_PROMPTS = [
  'Implémente un algorithme Python avec FastAPI pour diagonaliser une matrice',
  'Quelle est la relation entre la conscience morale chez Kant et le libre arbitre ?',
  'Explique l\'intrication quantique des particules et la structure de l\'ADN',
];

export const PromptAnalyzer: React.FC<PromptAnalyzerProps> = ({
  onAnalyzePrompt,
  loading,
  detectedDomains,
  domainsMap,
}) => {
  const [promptText, setPromptText] = useState<string>('');
  const [showWarning, setShowWarning] = useState<boolean>(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!promptText.trim()) return;
    onAnalyzePrompt(promptText.trim());
  };

  return (
    <div
      data-testid="prompt-analyzer"
      style={{
        marginTop: '18px',
        marginBottom: '20px',
        padding: '16px',
        backgroundColor: '#f8fafc',
        border: '1px solid #cbd5e1',
        borderRadius: '8px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', flexWrap: 'wrap', gap: '8px' }}>
        <label htmlFor="custom-prompt-input" style={{ fontWeight: 'bold', color: '#1e293b' }}>
          💬 Analyse sur Prompt Personnalisé (Routage Gating Network MoE) :
        </label>
        <button
          type="button"
          onClick={() => setShowWarning((prev) => !prev)}
          style={{
            backgroundColor: '#fef3c7',
            color: '#92400e',
            border: '1px solid #f59e0b',
            borderRadius: '4px',
            padding: '4px 10px',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          ⚠️ Prévention & Calibrage des données
        </button>
      </div>

      {showWarning && (
        <div
          role="alert"
          style={{
            marginBottom: '12px',
            padding: '10px 12px',
            backgroundColor: '#fffbeb',
            borderLeft: '4px solid #f59e0b',
            color: '#78350f',
            fontSize: '13px',
            borderRadius: '4px',
          }}
        >
          <strong>Note de prévention :</strong> L'analyse de routage MoE sur des prompts trop courts, bruités ou hors distribution peut biaiser la distribution Top-K des experts. Privilégiez des énoncés complets et représentatifs du domaine cible.
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <input
          id="custom-prompt-input"
          type="text"
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          placeholder="Saisissez un prompt libre (ex: Écris un algorithme Python pour résoudre une équation différentielle)..."
          style={{
            flex: 1,
            minWidth: '260px',
            padding: '9px 12px',
            fontSize: '14px',
            borderRadius: '6px',
            border: '1px solid #94a3b8',
          }}
        />
        <button
          type="submit"
          disabled={loading || !promptText.trim()}
          style={{
            padding: '9px 16px',
            backgroundColor: loading || !promptText.trim() ? '#94a3b8' : '#2563eb',
            color: '#ffffff',
            border: 'none',
            borderRadius: '6px',
            fontWeight: 'bold',
            cursor: loading || !promptText.trim() ? 'not-allowed' : 'pointer',
          }}
        >
          Analyser le Prompt
        </button>
      </form>

      <div style={{ marginTop: '10px', display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: '12px', color: '#64748b' }}>Exemples :</span>
        {SAMPLE_PROMPTS.map((sample, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => {
              setPromptText(sample);
              onAnalyzePrompt(sample);
            }}
            style={{
              fontSize: '12px',
              padding: '4px 8px',
              backgroundColor: '#e2e8f0',
              color: '#334155',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Exemple #{idx + 1}
          </button>
        ))}
      </div>

      {detectedDomains && Object.keys(detectedDomains).length > 0 && (
        <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#334155' }}>
            Pondération multi-domaines détectée :
          </span>
          {Object.entries(detectedDomains).map(([domId, pct]) => (
            <span
              key={domId}
              style={{
                fontSize: '12px',
                padding: '3px 8px',
                backgroundColor: '#dbeafe',
                color: '#1e40af',
                borderRadius: '12px',
                fontWeight: 'bold',
              }}
            >
              {domainsMap[domId] || domId} : {pct}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
};
