import React, { useState } from 'react';
import { TokenRoutingInfo } from '../types';

export interface TokenRoutingVisualizerProps {
  tokenRouting?: TokenRoutingInfo[];
  numLayers?: number;
  truncated?: boolean;
  originalTokenCount?: number;
  analyzedTokenCount?: number;
  isRealInference?: boolean;
}

const EXPERT_PALETTE = [
  '#2563eb', // Bleu
  '#7c3aed', // Violet
  '#db2777', // Rose fuchsia
  '#ea580c', // Orange vif
  '#16a34a', // Vert émeraude
  '#0891b2', // Cyan
  '#d97706', // Ambre doré
  '#4f46e5', // Indigo
  '#059669', // Vert jade
  '#dc2626', // Rouge brique
  '#9333ea', // Pourpre
  '#0284c7', // Bleu azur
];

export function getExpertColor(expertId: number): string {
  return EXPERT_PALETTE[Math.abs(expertId) % EXPERT_PALETTE.length];
}

export const TokenRoutingVisualizer: React.FC<TokenRoutingVisualizerProps> = ({
  tokenRouting,
  numLayers,
  truncated,
  originalTokenCount,
  analyzedTokenCount,
  isRealInference,
}) => {
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (!tokenRouting || tokenRouting.length === 0) {
    return null;
  }

  // Déterminer le nombre effectif de couches disponibles
  const detectedLayers = tokenRouting[0]?.layers?.length;
  const totalLayers = Math.max(numLayers || 0, detectedLayers || 1);
  const layerOptions = Array.from({ length: totalLayers }, (_, i) => i);

  return (
    <div
      data-testid="token-routing-visualizer"
      style={{
        marginTop: '16px',
        paddingTop: '14px',
        borderTop: '1px dashed #cbd5e1',
      }}
    >
      {/* 1. Alerte visuelle de troncature si applicable */}
      {truncated && (
        <div
          role="alert"
          data-testid="truncation-warning"
          style={{
            marginBottom: '14px',
            padding: '10px 14px',
            backgroundColor: '#fffbeb',
            border: '1px solid #fde68a',
            borderLeft: '5px solid #d97706',
            borderRadius: '6px',
            color: '#92400e',
            fontSize: '13px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <span style={{ fontSize: '18px' }}>⚠️</span>
          <div>
            <strong>Prompt tronqué à l'analyse :</strong> Seuls les{' '}
            <strong>{analyzedTokenCount ?? tokenRouting.length}</strong> premiers tokens sur{' '}
            <strong>{originalTokenCount ?? tokenRouting.length}</strong> tokens détectés ont été analysés par le modèle MoE (limite maximale : 128 tokens). Les tokens au-delà ne sont pas routés.
          </div>
        </div>
      )}

      {/* 2. En-tête et sélecteur de couche MoE */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          marginBottom: '12px',
        }}
      >
        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#1e293b' }}>
          🔬 Routage par Token sur toutes les couches{' '}
          <span style={{ fontSize: '11px', color: '#64748b', fontWeight: 'normal' }}>
            ({tokenRouting.length} tokens analysés {isRealInference ? '— PyTorch réel' : '— Simulation'})
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            data-testid="layer-tabs"
            style={{
              display: 'flex',
              gap: '6px',
              flexWrap: 'wrap',
              alignItems: 'center',
            }}
          >
            <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#475569', marginRight: '4px' }}>
              Couche :
            </span>
            {layerOptions.map((layerIdx) => {
              const isSelected = selectedLayer === layerIdx;
              return (
                <button
                  key={layerIdx}
                  type="button"
                  data-testid={`layer-tab-${layerIdx}`}
                  onClick={() => setSelectedLayer(layerIdx)}
                  style={{
                    padding: '4px 12px',
                    fontSize: '12px',
                    fontWeight: isSelected ? 'bold' : '500',
                    borderRadius: '6px',
                    border: isSelected ? '1px solid #2563eb' : '1px solid #cbd5e1',
                    backgroundColor: isSelected ? '#2563eb' : '#ffffff',
                    color: isSelected ? '#ffffff' : '#334155',
                    cursor: 'pointer',
                    boxShadow: isSelected ? '0 1px 3px rgba(37,99,235,0.3)' : 'none',
                    transition: 'all 0.15s ease',
                  }}
                >
                  Couche {layerIdx}
                </button>
              );
            })}
          </div>

          <select
            id="layer-selector"
            data-testid="layer-selector"
            aria-label="Sélectionner une couche"
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(Number(e.target.value))}
            style={{
              display: totalLayers > 8 ? 'inline-block' : 'none',
              padding: '4px 8px',
              fontSize: '12px',
              borderRadius: '6px',
              border: '1px solid #94a3b8',
              backgroundColor: '#ffffff',
            }}
          >
            {layerOptions.map((layerIdx) => (
              <option key={layerIdx} value={layerIdx}>
                Couche {layerIdx}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 3. Affichage interactif des tokens */}
      <div
        data-testid="token-list"
        style={{
          display: 'flex',
          gap: '8px',
          flexWrap: 'wrap',
          position: 'relative',
        }}
      >
        {tokenRouting.map((tr, idx) => {
          // Extraire les informations de routage pour la couche sélectionnée
          const layerData = tr.layers?.find((l) => l.layer_id === selectedLayer) ?? (
            selectedLayer === 0
              ? {
                  layer_id: 0,
                  expert_id: tr.expert_id,
                  confidence: tr.confidence,
                  top_k: tr.top_k || [{ expert_id: tr.expert_id, weight: tr.confidence }],
                }
              : undefined
          );

          const dominantExpert = layerData ? layerData.expert_id : tr.expert_id;
          const confidence = layerData ? layerData.confidence : tr.confidence;
          const topKList = layerData?.top_k || (tr.top_k && selectedLayer === 0 ? tr.top_k : []);
          const expertColor = getExpertColor(dominantExpert);
          const isHovered = hoveredIndex === idx;

          return (
            <div
              key={`${tr.token}-${idx}`}
              data-testid="token-item"
              onMouseEnter={() => setHoveredIndex(idx)}
              onMouseLeave={() => setHoveredIndex(null)}
              onFocus={() => setHoveredIndex(idx)}
              onBlur={() => setHoveredIndex(null)}
              tabIndex={0}
              style={{
                position: 'relative',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                padding: '4px 8px',
                backgroundColor: isHovered ? '#1e293b' : '#0f172a',
                color: '#f8fafc',
                borderRadius: '6px',
                border: isHovered ? `2px solid ${expertColor}` : '1px solid #334155',
                cursor: 'pointer',
                fontFamily: 'monospace',
                fontSize: '12px',
                outline: 'none',
                boxShadow: isHovered ? '0 4px 10px rgba(0, 0, 0, 0.25)' : 'none',
                transition: 'border 0.15s ease, background-color 0.15s ease',
              }}
            >
              <span style={{ fontWeight: '600' }}>{tr.token}</span>

              {/* Badge de l'expert dominant */}
              <span
                data-testid="dominant-expert-badge"
                style={{
                  backgroundColor: expertColor,
                  color: '#ffffff',
                  padding: '1px 5px',
                  borderRadius: '3px',
                  fontSize: '10px',
                  fontWeight: 'bold',
                }}
              >
                E{dominantExpert}
              </span>

              {/* Pourcentage de confiance */}
              <span style={{ color: '#34d399', fontSize: '11px' }}>
                {confidence}%
              </span>

              {/* 4. Infobulle au survol détaillant le Top-K de cette couche */}
              {isHovered && (
                <div
                  role="tooltip"
                  data-testid="token-tooltip"
                  style={{
                    position: 'absolute',
                    bottom: 'calc(100% + 8px)',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    zIndex: 50,
                    minWidth: '190px',
                    maxWidth: '280px',
                    backgroundColor: '#0f172a',
                    color: '#f8fafc',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.45)',
                    border: '1px solid #475569',
                    fontSize: '12px',
                    pointerEvents: 'none',
                    textAlign: 'left',
                  }}
                >
                  <div
                    style={{
                      borderBottom: '1px solid #334155',
                      paddingBottom: '5px',
                      marginBottom: '6px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <span style={{ fontWeight: 'bold', color: '#38bdf8' }}>
                      Token #{idx + 1} : « {tr.token} »
                    </span>
                    <span style={{ fontSize: '11px', color: '#94a3b8' }}>
                      Couche {selectedLayer}
                    </span>
                  </div>

                  <div style={{ marginBottom: '6px', fontSize: '11px', color: '#cbd5e1' }}>
                    Expert dominant :{' '}
                    <strong style={{ color: expertColor }}>
                      Expert {dominantExpert}
                    </strong>{' '}
                    ({confidence}%)
                  </div>

                  {topKList.length > 0 ? (
                    <div>
                      <div
                        style={{
                          fontSize: '11px',
                          fontWeight: 'bold',
                          color: '#94a3b8',
                          marginBottom: '4px',
                          textTransform: 'uppercase',
                          letterSpacing: '0.5px',
                        }}
                      >
                        Top-{topKList.length} Gating Network :
                      </div>
                      <div
                        data-testid="top-k-list"
                        style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}
                      >
                        {topKList.map((exp, rank) => {
                          const c = getExpertColor(exp.expert_id);
                          return (
                            <div
                              key={exp.expert_id}
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                fontSize: '11px',
                                gap: '8px',
                              }}
                            >
                              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                <span style={{ color: '#94a3b8', fontSize: '10px' }}>
                                  #{rank + 1}
                                </span>
                                <span
                                  style={{
                                    display: 'inline-block',
                                    width: '8px',
                                    height: '8px',
                                    borderRadius: '50%',
                                    backgroundColor: c,
                                  }}
                                />
                                <span style={{ color: '#f1f5f9', fontWeight: 'bold' }}>
                                  Expert {exp.expert_id}
                                </span>
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <div
                                  style={{
                                    width: '40px',
                                    height: '5px',
                                    backgroundColor: '#334155',
                                    borderRadius: '3px',
                                    overflow: 'hidden',
                                  }}
                                >
                                  <div
                                    style={{
                                      width: `${Math.min(100, exp.weight)}%`,
                                      height: '100%',
                                      backgroundColor: c,
                                    }}
                                  />
                                </div>
                                <span style={{ color: '#34d399', fontWeight: '600', minWidth: '36px', textAlign: 'right' }}>
                                  {exp.weight}%
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div style={{ fontSize: '11px', color: '#94a3b8' }}>
                      Aucun détail Top-K disponible pour cette couche.
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
