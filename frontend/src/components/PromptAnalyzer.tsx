import React, { useState, useEffect } from 'react';
import { DomainsMap, TokenRoutingInfo } from '../types';
import { fetchPrompts, PromptOption } from '../api';

interface PromptAnalyzerProps {
  onAnalyzePrompt: (prompt: string) => void;
  loading: boolean;
  detectedDomains?: { [domainId: string]: number };
  domainsMap: DomainsMap;
  tokenRouting?: TokenRoutingInfo[];
  isRealInference?: boolean;
}

export const PromptAnalyzer: React.FC<PromptAnalyzerProps> = ({
  onAnalyzePrompt,
  loading,
  detectedDomains,
  domainsMap,
  tokenRouting,
  isRealInference,
}) => {
  const [promptText, setPromptText] = useState<string>('');
  const [selectedPromptId, setSelectedPromptId] = useState<string>('');
  const [prompts, setPrompts] = useState<PromptOption[]>([]);
  const [showWarning, setShowWarning] = useState<boolean>(false);

  useEffect(() => {
    fetchPrompts().then(data => {
      setPrompts(data);
    }).catch(err => {
      console.error(err);
    });
  }, []);

  const handleSelectPrompt = (id: string) => {
    setSelectedPromptId(id);
    if (id) {
      setPromptText(id);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const targetPrompt = promptText.trim() || selectedPromptId.trim();
    if (!targetPrompt) return;
    onAnalyzePrompt(targetPrompt);
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

      {prompts.length > 0 && (
        <div style={{ marginBottom: '10px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <label htmlFor="prompt-selector" style={{ fontSize: '13px', fontWeight: 'bold', color: '#475569' }}>
            📂 Charger un prompt prédéfini (.md) :
          </label>
          <select
            id="prompt-selector"
            data-testid="prompt-selector"
            value={selectedPromptId}
            onChange={(e) => handleSelectPrompt(e.target.value)}
            style={{
              padding: '6px 10px',
              fontSize: '13px',
              borderRadius: '6px',
              border: '1px solid #94a3b8',
              backgroundColor: '#ffffff',
            }}
          >
            <option value="">-- Choisir un prompt existant --</option>
            {prompts.map(p => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <input
          id="custom-prompt-input"
          data-testid="custom-prompt-input"
          type="text"
          value={promptText}
          onChange={(e) => {
            setPromptText(e.target.value);
            if (selectedPromptId && e.target.value !== selectedPromptId) {
              setSelectedPromptId('');
            }
          }}
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
          disabled={loading || (!promptText.trim() && !selectedPromptId.trim())}
          style={{
            padding: '9px 16px',
            backgroundColor: loading || (!promptText.trim() && !selectedPromptId.trim()) ? '#94a3b8' : '#2563eb',
            color: '#ffffff',
            border: 'none',
            borderRadius: '6px',
            fontWeight: 'bold',
            cursor: loading || (!promptText.trim() && !selectedPromptId.trim()) ? 'not-allowed' : 'pointer',
          }}
        >
          Analyser le Prompt
        </button>
      </form>

      {detectedDomains && Object.keys(detectedDomains).length > 0 && (
        <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#334155' }}>
            Pondération sémantique détectée :
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

      {tokenRouting && tokenRouting.length > 0 && (
        <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px dashed #cbd5e1' }}>
          <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#334155', marginBottom: '6px' }}>
            🔬 Routage Gating Network par Token (Couche 0 {isRealInference ? '— Tenseur PyTorch réel' : '— Simulation Top-K'}) :
          </div>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {tokenRouting.map((tr, i) => (
              <span
                key={`${tr.token}-${i}`}
                style={{
                  fontSize: '11px',
                  padding: '3px 7px',
                  backgroundColor: '#0f172a',
                  color: '#f8fafc',
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                }}
              >
                {tr.token} <span style={{ color: '#38bdf8' }}>→ E{tr.expert_id}</span>{' '}
                <span style={{ color: '#34d399' }}>({tr.confidence}%)</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
