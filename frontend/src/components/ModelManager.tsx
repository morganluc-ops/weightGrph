import React, { useState, useEffect } from 'react';
import { MoEModelOption, ModelInfo } from '../types';

interface ModelManagerProps {
  models: MoEModelOption[];
  currentModel: ModelInfo | null;
  onSelectModel: (modelId: string) => void;
  loadingModel: boolean;
  loadingTargetId?: string | null;
}

export const ModelManager: React.FC<ModelManagerProps> = ({
  models,
  currentModel,
  onSelectModel,
  loadingModel,
  loadingTargetId,
}) => {
  const [selectedId, setSelectedId] = useState<string>(
    currentModel?.model_id || (models[0] ? models[0].id : '')
  );
  const [customHuggingFaceId, setCustomHuggingFaceId] = useState<string>('');
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);

  useEffect(() => {
    if (currentModel?.model_id) {
      setSelectedId(currentModel.model_id);
    }
  }, [currentModel?.model_id]);

  useEffect(() => {
    let interval: any = null;
    if (loadingModel) {
      setElapsedSeconds(0);
      interval = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      setElapsedSeconds(0);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [loadingModel]);

  const handleLoadSelected = () => {
    const target = customHuggingFaceId.trim() || selectedId;
    if (target) {
      onSelectModel(target);
      setCustomHuggingFaceId('');
    }
  };

  const selectedModelObj = models.find((m) => m.id === selectedId);

  return (
    <div
      data-testid="model-manager"
      style={{
        marginBottom: '20px',
        padding: '16px',
        backgroundColor: '#ecfdf5',
        border: '1px solid #10b981',
        borderRadius: '8px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
        <div style={{ fontWeight: 'bold', color: '#0f172a', fontSize: '15px' }}>
          ⚙️ Modèle MoE PyTorch Actif
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
          🟢 {currentModel?.model_id || 'Modèle actif'} ({currentModel?.device?.toUpperCase() || 'CPU'}) — Inférence réelle
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
            padding: '8px 18px',
            backgroundColor: loadingModel ? '#94a3b8' : '#0f766e',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            fontWeight: 'bold',
            cursor: loadingModel ? 'wait' : 'pointer',
          }}
        >
          {loadingModel ? '⏳ Chargement en cours...' : 'Charger le Modèle MoE'}
        </button>
      </div>

      {loadingModel && (
        <div
          role="status"
          style={{
            marginTop: '12px',
            padding: '12px 14px',
            backgroundColor: '#eff6ff',
            border: '1px solid #3b82f6',
            borderRadius: '6px',
            color: '#1e3a8a',
            fontSize: '13px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <strong>
              📥 Téléchargement / Chargement PyTorch en cours : {loadingTargetId || selectedId}
            </strong>
            <span style={{ fontWeight: 'bold', fontFamily: 'monospace' }}>
              Temps écoulé : {elapsedSeconds}s
            </span>
          </div>
          <div style={{ color: '#475569', fontSize: '12px' }}>
            Poids estimés : {selectedModelObj?.size_hint || 'selon le modèle'}. Les tenseurs PyTorch sont téléchargés et montés en RAM/VRAM... Les analyses précédentes ont été réinitialisées.
          </div>
        </div>
      )}
    </div>
  );
};
