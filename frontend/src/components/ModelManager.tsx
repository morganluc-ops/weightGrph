import React, { useState, useEffect } from 'react';
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
  const [selectedId, setSelectedId] = useState<string>(
    currentModel?.model_id || (models[0] ? models[0].id : '')
  );
  const [customHuggingFaceId, setCustomHuggingFaceId] = useState<string>('');

  useEffect(() => {
    if (currentModel?.model_id) {
      setSelectedId(currentModel.model_id);
    }
  }, [currentModel?.model_id]);

  const handleLoadSelected = () => {
    const target = customHuggingFaceId.trim() || selectedId;
    if (target) {
      onSelectModel(target);
      setCustomHuggingFaceId('');
    }
  };

  return (
    <div
      data-testid="model-manager"
      style={{
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: '#ecfdf5',
        border: '1px solid #10b981',
        borderRadius: '8px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px', marginBottom: '10px' }}>
        <div style={{ fontWeight: 'bold', color: '#0f172a', fontSize: '15px' }}>
          ⚙️ Modèle MoE PyTorch / HuggingFace Actif
        </div>
        <span
          style={{
            padding: '4px 10px',
            borderRadius: '12px',
            fontSize: '12px',
            fontWeight: 'bold',
            backgroundColor: '#059669',
            color: '#ffffff',
          }}
        >
          🟢 Inférence PyTorch Réelle ({currentModel?.device?.toUpperCase() || 'CPU'}) — Extraction directe de router_logits
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
            border: '1px solid #10b981',
            backgroundColor: '#fff',
            minWidth: '320px',
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
          placeholder="Ou saisir n'importe quel ID HuggingFace MoE (ex: google/switch-base-8)..."
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
          {loadingModel ? '⏳ Téléchargement / Chargement PyTorch...' : 'Charger le Modèle MoE'}
        </button>
      </div>
    </div>
  );
};
