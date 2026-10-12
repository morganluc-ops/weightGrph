import React, { useState } from 'react';
import { MoEModelOption, ModelInfo } from '../types';

interface ModelManagerProps {
  models: MoEModelOption[];
  currentModel: ModelInfo | null;
  onSelectModel: (modelId: string) => void;
  loadingModel: boolean;
}

export const ModelManager: React.FC<ModelManagerProps> = ({
  models,
  currentModel,
  onSelectModel,
  loadingModel,
}) => {
  const [selectedId, setSelectedId] = useState<string>(currentModel?.model_id || 'simulated');
  const [customHuggingFaceId, setCustomHuggingFaceId] = useState<string>('');

  const handleLoadSelected = () => {
    const target = customHuggingFaceId.trim() || selectedId;
    if (target) {
      onSelectModel(target);
      setCustomHuggingFaceId('');
    }
  };

  const isReal = Boolean(currentModel?.is_real_model);

  return (
    <div
      data-testid="model-manager"
      style={{
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: isReal ? '#ecfdf5' : '#f1f5f9',
        border: `1px solid ${isReal ? '#10b981' : '#cbd5e1'}`,
        borderRadius: '8px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px', marginBottom: '10px' }}>
        <div style={{ fontWeight: 'bold', color: '#0f172a', fontSize: '15px' }}>
          ⚙️ Moteur MoE & Gestionnaire de Modèles (HuggingFace / PyTorch)
        </div>
        <span
          style={{
            padding: '4px 10px',
            borderRadius: '12px',
            fontSize: '12px',
            fontWeight: 'bold',
            backgroundColor: isReal ? '#059669' : '#475569',
            color: '#ffffff',
          }}
        >
          {isReal
            ? `🟢 Vrai Modèle MoE Actif (${currentModel?.device?.toUpperCase() || 'CPU'}) — router_logits réels`
            : '⚡ Mode Simulation Dynamique Token-Level (Sans téléchargement)'}
        </span>
      </div>

      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'center' }}>
        <select
          aria-label="Sélectionner un modèle MoE"
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
          disabled={loadingModel}
          style={{
            padding: '8px 12px',
            fontSize: '14px',
            borderRadius: '6px',
            border: '1px solid #94a3b8',
            backgroundColor: '#fff',
            minWidth: '300px',
          }}
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} [{m.size_hint}]
            </option>
          ))}
        </select>

        <input
          type="text"
          value={customHuggingFaceId}
          onChange={(e) => setCustomHuggingFaceId(e.target.value)}
          placeholder="Ou ID HuggingFace personnalisé (ex: google/switch-base-8)..."
          disabled={loadingModel}
          style={{
            flex: 1,
            minWidth: '220px',
            padding: '8px 12px',
            fontSize: '13px',
            borderRadius: '6px',
            border: '1px solid #94a3b8',
          }}
        />

        <button
          type="button"
          onClick={handleLoadSelected}
          disabled={loadingModel}
          style={{
            padding: '8px 16px',
            backgroundColor: loadingModel ? '#94a3b8' : '#0f766e',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            fontWeight: 'bold',
            cursor: loadingModel ? 'wait' : 'pointer',
          }}
        >
          {loadingModel ? '⏳ Chargement du modèle MoE...' : 'Charger le Modèle'}
        </button>
      </div>
    </div>
  );
};
