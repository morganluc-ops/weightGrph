import React, { useState, useEffect, useRef } from 'react';
import { DomainsMap, TokenRoutingInfo } from '../types';
import { fetchPrompts, uploadPromptFile, PromptOption } from '../api';

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
  const [selectedPromptId, setSelectedPromptId] = useState<string>('');
  const [prompts, setPrompts] = useState<PromptOption[]>([]);
  const [showWarning, setShowWarning] = useState<boolean>(false);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    fetchPrompts()
      .then((data) => {
        setPrompts(data);
      })
      .catch((err) => {
        console.error(err);
      });
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.md')) {
      setUploadError('Seuls les fichiers .md sont autorisés');
      return;
    }

    try {
      setUploading(true);
      setUploadError(null);
      const newPrompt = await uploadPromptFile(file);
      setPrompts((prev) => {
        const exists = prev.some((p) => p.id === newPrompt.id);
        if (exists) return prev;
        return [...prev, newPrompt].sort((a, b) => a.name.localeCompare(b.name));
      });
      setSelectedPromptId(newPrompt.id);
    } catch (err: any) {
      setUploadError(err.message || "Erreur lors de l'upload du fichier");
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPromptId.trim()) return;
    onAnalyzePrompt(selectedPromptId.trim());
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
        <label htmlFor="prompt-selector" style={{ fontWeight: 'bold', color: '#1e293b' }}>
          💬 Analyse sur Prompt (.md) (Routage Gating Network MoE) :
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

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          ref={fileInputRef}
          type="file"
          accept=".md,.MD"
          data-testid="upload-prompt-input"
          style={{ display: 'none' }}
          onChange={handleFileUpload}
        />

        <button
          type="button"
          data-testid="upload-prompt-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading || loading}
          style={{
            padding: '9px 14px',
            backgroundColor: uploading ? '#cbd5e1' : '#0f766e',
            color: '#ffffff',
            border: 'none',
            borderRadius: '6px',
            fontWeight: '600',
            fontSize: '13px',
            cursor: uploading || loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          {uploading ? '⏳ Téléversement...' : '📤 Uploader un prompt .md'}
        </button>

        <select
          id="prompt-selector"
          data-testid="prompt-selector"
          value={selectedPromptId}
          onChange={(e) => setSelectedPromptId(e.target.value)}
          style={{
            flex: 1,
            minWidth: '240px',
            padding: '9px 12px',
            fontSize: '14px',
            borderRadius: '6px',
            border: '1px solid #94a3b8',
            backgroundColor: '#ffffff',
            color: selectedPromptId ? '#0f172a' : '#64748b',
          }}
        >
          <option value="">-- Sélectionner un fichier de prompt .md --</option>
          {prompts.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} ({p.id})
            </option>
          ))}
        </select>

        <button
          type="submit"
          data-testid="analyze-prompt-btn"
          disabled={loading || uploading || !selectedPromptId.trim()}
          style={{
            padding: '9px 18px',
            backgroundColor: loading || uploading || !selectedPromptId.trim() ? '#94a3b8' : '#2563eb',
            color: '#ffffff',
            border: 'none',
            borderRadius: '6px',
            fontWeight: 'bold',
            fontSize: '14px',
            cursor: loading || uploading || !selectedPromptId.trim() ? 'not-allowed' : 'pointer',
          }}
        >
          Analyser le Prompt
        </button>
      </form>

      {uploadError && (
        <div data-testid="upload-error" style={{ color: '#dc2626', fontSize: '13px', marginTop: '8px' }}>
          ⚠️ {uploadError}
        </div>
      )}

      {selectedPromptId && (
        <div style={{ marginTop: '8px', fontSize: '12px', color: '#64748b' }}>
          Fichier sélectionné : <span style={{ fontFamily: 'monospace', color: '#0f172a', fontWeight: 'bold' }}>{selectedPromptId}</span>
        </div>
      )}

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
