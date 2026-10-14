import React, { useState, useEffect } from 'react';
import { MoEModelOption, ModelInfo, DownloadProgressInfo } from '../types';
import { fetchModelProgress } from '../api';

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
  const [progressInfo, setProgressInfo] = useState<DownloadProgressInfo | null>(null);

  useEffect(() => {
    if (currentModel?.model_id) {
      setSelectedId(currentModel.model_id);
    }
  }, [currentModel?.model_id]);

  useEffect(() => {
    let interval: any = null;
    let pollInterval: any = null;

    if (loadingModel) {
      setElapsedSeconds(0);
      interval = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);

      const checkProgress = async () => {
        try {
          const prog = await fetchModelProgress();
          setProgressInfo(prog);
        } catch {
          // ignore transient poll error
        }
      };

      checkProgress();
      pollInterval = setInterval(checkProgress, 400);
    } else {
      setElapsedSeconds(0);
      setProgressInfo(null);
    }

    return () => {
      if (interval) clearInterval(interval);
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [loadingModel]);

  const handleLoadSelected = () => {
    const target = customHuggingFaceId.trim() || selectedId;
    if (target) {
      onSelectModel(target);
      setCustomHuggingFaceId('');
    }
  };

  const folderInputRef = React.useRef<HTMLInputElement>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleFolderSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const first = files[0];
    const fileWithPath = first as File & { path?: string };

    if (fileWithPath.path) {
      const fullPath = fileWithPath.path;
      if (first.webkitRelativePath) {
        const relParts = first.webkitRelativePath.split('/');
        const rootDirName = relParts[0];
        const normalized = fullPath.replace(/\\/g, '/');
        const rootIndex = normalized.lastIndexOf(rootDirName);
        if (rootIndex !== -1) {
          setCustomHuggingFaceId(fullPath.substring(0, rootIndex + rootDirName.length));
          return;
        }
      }
      setCustomHuggingFaceId(fullPath);
    } else if (first.webkitRelativePath) {
      const folderName = first.webkitRelativePath.split('/')[0];
      setCustomHuggingFaceId(folderName);
    } else if (first.name) {
      setCustomHuggingFaceId(first.name);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const first = files[0];
    const fileWithPath = first as File & { path?: string };
    if (fileWithPath.path) {
      setCustomHuggingFaceId(fileWithPath.path);
    } else {
      setCustomHuggingFaceId(first.name);
    }
  };

  const selectedModelObj = models.find((m) => m.id === selectedId);
  const displayPercent = progressInfo?.progress_percent ?? (loadingModel ? 5 : 0);

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
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span
            data-testid="active-model-badge"
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
          <div
            role="button"
            tabIndex={0}
            data-testid="model-help-tooltip"
            aria-label="Aide sur les modèles MoE"
            title="1. Placez les dossiers du modèle sur le serveur ou en local. 2. Entrez le chemin absolu vers le dossier (ex: /chemin/vers/le/modele) ou sélectionnez un dossier local."
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '22px',
              height: '22px',
              borderRadius: '50%',
              backgroundColor: '#e2e8f0',
              color: '#475569',
              fontWeight: 'bold',
              fontSize: '13px',
              cursor: 'help',
              userSelect: 'none',
              boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
            }}
          >
            ?
          </div>
        </div>
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
            minWidth: '280px',
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
          placeholder="Ou ID HuggingFace personnalisé / chemin local..."
          disabled={loadingModel}
          data-testid="custom-model-input"
          style={{
            flex: 1,
            minWidth: '240px',
            padding: '8px 12px',
            fontSize: '13px',
            borderRadius: '6px',
            border: '1px solid #94a3b8',
          }}
        />

        <input
          ref={folderInputRef}
          type="file"
          // @ts-expect-error webkitdirectory is supported by Chromium/WebKit browsers
          webkitdirectory=""
          directory=""
          multiple
          style={{ display: 'none' }}
          onChange={handleFolderSelect}
          data-testid="local-folder-input"
        />

        <input
          ref={fileInputRef}
          type="file"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
          data-testid="local-file-input"
        />

        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          <button
            type="button"
            onClick={() => folderInputRef.current?.click()}
            disabled={loadingModel}
            title="Sélectionner un dossier local contenant les poids du modèle MoE"
            aria-label="Sélectionner un dossier local"
            data-testid="browse-folder-button"
            style={{
              padding: '8px 12px',
              backgroundColor: '#f8fafc',
              color: '#334155',
              border: '1px solid #cbd5e1',
              borderRadius: '6px',
              fontSize: '13px',
              fontWeight: 500,
              cursor: loadingModel ? 'not-allowed' : 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              whiteSpace: 'nowrap',
            }}
          >
            📁 Parcourir dossier...
          </button>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={loadingModel}
            title="Sélectionner un fichier local (.safetensors, .pt, .json)"
            aria-label="Sélectionner un fichier local"
            data-testid="browse-file-button"
            style={{
              padding: '8px 12px',
              backgroundColor: '#f8fafc',
              color: '#334155',
              border: '1px solid #cbd5e1',
              borderRadius: '6px',
              fontSize: '13px',
              fontWeight: 500,
              cursor: loadingModel ? 'not-allowed' : 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              whiteSpace: 'nowrap',
            }}
          >
            📄 Fichier...
          </button>
        </div>

        <button
          type="button"
          onClick={handleLoadSelected}
          disabled={loadingModel}
          data-testid="load-model-button"
          style={{
            padding: '8px 18px',
            backgroundColor: loadingModel ? '#94a3b8' : '#0f766e',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            fontWeight: 'bold',
            cursor: loadingModel ? 'wait' : 'pointer',
            whiteSpace: 'nowrap',
          }}
        >
          {loadingModel ? '⏳ Téléchargement en cours...' : 'Charger le Modèle MoE'}
        </button>
      </div>

      {loadingModel && (
        <div
          role="status"
          style={{
            marginTop: '14px',
            padding: '14px 16px',
            backgroundColor: '#eff6ff',
            border: '1px solid #3b82f6',
            borderRadius: '8px',
            color: '#1e3a8a',
            fontSize: '13px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '6px' }}>
            <span style={{ fontWeight: 'bold', fontSize: '14px' }}>
              📥 {loadingTargetId || selectedId}
            </span>
            <div style={{ display: 'flex', gap: '12px', fontSize: '12px', fontFamily: 'monospace' }}>
              {progressInfo?.speed_mb_s ? (
                <span style={{ color: '#2563eb', fontWeight: 'bold' }}>
                  ⚡ {progressInfo.speed_mb_s} Mo/s
                </span>
              ) : null}
              <span style={{ color: '#475569', fontWeight: 'bold' }}>
                ⏱️ {elapsedSeconds}s écoulées
              </span>
            </div>
          </div>

          {/* Real Animated Progress Bar */}
          <div
            style={{
              width: '100%',
              height: '18px',
              backgroundColor: '#dbeafe',
              borderRadius: '9px',
              overflow: 'hidden',
              position: 'relative',
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.1)',
            }}
          >
            <div
              style={{
                width: `${Math.min(100, Math.max(3, displayPercent))}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #3b82f6 0%, #10b981 100%)',
                transition: 'width 0.3s ease-out',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                paddingRight: '6px',
              }}
            />
            <span
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                fontSize: '11px',
                fontWeight: 'bold',
                color: displayPercent > 55 ? '#ffffff' : '#1e3a8a',
              }}
            >
              {displayPercent.toFixed(1)} %
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '12px', color: '#475569', flexWrap: 'wrap', gap: '6px' }}>
            <span>
              <strong>Fichier :</strong> {progressInfo?.current_file || 'Vérification du cache HuggingFace...'}
            </span>
            <span>
              {progressInfo?.downloaded_mb && progressInfo?.total_mb
                ? `${progressInfo.downloaded_mb} Mo / ${progressInfo.total_mb} Mo`
                : `Taille estimée : ${selectedModelObj?.size_hint || '1.2 Go'}`}
            </span>
          </div>

          <div style={{ marginTop: '4px', fontSize: '12px', color: '#1d4ed8', fontStyle: 'italic' }}>
            {progressInfo?.stage_message || 'Téléchargement des poids PyTorch en cours...'}
          </div>
        </div>
      )}
    </div>
  );
};
