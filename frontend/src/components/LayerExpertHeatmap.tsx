import React, { useState } from 'react';
import { DomainAnalysisResult } from '../types';

interface LayerExpertHeatmapProps {
  analysis: DomainAnalysisResult | null;
  numLayers?: number;
  expertsPerLayer?: number;
}

interface HoveredCellInfo {
  layer: number;
  expert: number;
  layerVal: number;
  globalVal: number;
  isDead: boolean;
  x?: number;
  y?: number;
}

export const LayerExpertHeatmap: React.FC<LayerExpertHeatmapProps> = ({
  analysis,
  numLayers: propNumLayers,
  expertsPerLayer: propExpertsPerLayer,
}) => {
  const [normMode, setNormMode] = useState<'layer' | 'global'>('layer');
  const [showValues, setShowValues] = useState<boolean>(true);
  const [hoveredCell, setHoveredCell] = useState<HoveredCellInfo | null>(null);

  if (!analysis || !analysis.layer_expert_matrix || analysis.layer_expert_matrix.length === 0) {
    return null;
  }

  const matrix = analysis.layer_expert_matrix;
  const numLayers = matrix.length;
  const expertsPerLayer = matrix[0]?.length || 0;

  if (numLayers === 0 || expertsPerLayer === 0) {
    return null;
  }

  // Row maximums for layer normalization
  const rowMaxes = matrix.map((row) => Math.max(0.0001, ...row));
  // Global maximum across all cells
  const globalMax = Math.max(0.0001, ...matrix.flatMap((row) => row));

  // Compute stats across the whole matrix
  let totalDead = 0;
  let maxCell = { layer: 0, expert: 0, val: 0 };
  for (let l = 0; l < numLayers; l++) {
    for (let e = 0; e < expertsPerLayer; e++) {
      const v = matrix[l][e];
      if (v <= 0) totalDead++;
      if (v > maxCell.val) {
        maxCell = { layer: l, expert: e, val: v };
      }
    }
  }

  const activeCount = numLayers * expertsPerLayer - totalDead;

  // Color generator: light blue/indigo to dark violet/indigo
  const getCellColor = (val: number, layerIdx: number): { bg: string; text: string; isDead: boolean } => {
    if (val <= 0) {
      return { bg: '#f1f5f9', text: '#94a3b8', isDead: true };
    }

    const intensity =
      normMode === 'layer'
        ? Math.min(1, Math.max(0.05, val / rowMaxes[layerIdx]))
        : Math.min(1, Math.max(0.05, val / globalMax));

    // Gradient from soft indigo (light) to deep royal indigo (dark)
    // Low: rgb(224, 231, 255) -> High: rgb(49, 46, 129)
    const r = Math.round(224 - intensity * (224 - 49));
    const g = Math.round(231 - intensity * (231 - 46));
    const b = Math.round(255 - intensity * (255 - 129));

    const bg = `rgb(${r}, ${g}, ${b})`;
    // White text when cell is dark enough for contrast
    const text = intensity > 0.45 ? '#ffffff' : '#1e1b4b';

    return { bg, text, isDead: false };
  };

  const activeHover = hoveredCell || {
    layer: maxCell.layer,
    expert: maxCell.expert,
    layerVal: maxCell.val,
    globalVal: maxCell.val / numLayers,
    isDead: maxCell.val <= 0,
  };

  return (
    <div
      data-testid="layer-expert-heatmap"
      style={{
        marginTop: '20px',
        padding: '20px',
        backgroundColor: '#ffffff',
        borderRadius: '10px',
        border: '1px solid #e2e8f0',
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
      }}
    >
      {/* Header & Controls */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          marginBottom: '16px',
        }}
      >
        <div>
          <h3
            style={{
              margin: 0,
              fontSize: '18px',
              color: '#1e293b',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            🔥 Heatmap Couche × Expert du Routage MoE
          </h3>
          <span style={{ fontSize: '12px', color: '#64748b' }}>
            Distribution bidimensionnelle des activations entre couches neuronales et experts
          </span>
        </div>

        {/* View toggles */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
          <div
            style={{
              display: 'inline-flex',
              borderRadius: '6px',
              border: '1px solid #cbd5e1',
              overflow: 'hidden',
              backgroundColor: '#f8fafc',
            }}
          >
            <button
              type="button"
              data-testid="btn-norm-layer"
              onClick={() => setNormMode('layer')}
              style={{
                padding: '5px 12px',
                border: 'none',
                backgroundColor: normMode === 'layer' ? '#2563eb' : 'transparent',
                color: normMode === 'layer' ? '#ffffff' : '#475569',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              Normalisation par Couche
            </button>
            <button
              type="button"
              data-testid="btn-norm-global"
              onClick={() => setNormMode('global')}
              style={{
                padding: '5px 12px',
                border: 'none',
                backgroundColor: normMode === 'global' ? '#2563eb' : 'transparent',
                color: normMode === 'global' ? '#ffffff' : '#475569',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              Normalisation Globale
            </button>
          </div>

          <button
            type="button"
            data-testid="btn-toggle-values"
            onClick={() => setShowValues(!showValues)}
            style={{
              padding: '5px 12px',
              backgroundColor: showValues ? '#e0f2fe' : '#f1f5f9',
              border: '1px solid #cbd5e1',
              borderRadius: '6px',
              color: showValues ? '#0369a1' : '#475569',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            {showValues ? '🔢 Masquer valeurs' : '🔢 Afficher valeurs'}
          </button>
        </div>
      </div>

      {/* Quick summary stats bar */}
      <div
        style={{
          display: 'flex',
          gap: '16px',
          flexWrap: 'wrap',
          marginBottom: '16px',
          padding: '8px 12px',
          backgroundColor: '#f8fafc',
          borderRadius: '6px',
          border: '1px solid #e2e8f0',
          fontSize: '12px',
          color: '#334155',
        }}
      >
        <span>
          <strong>Architecture :</strong> {numLayers} couches × {expertsPerLayer} experts ({numLayers * expertsPerLayer} au total)
        </span>
        <span>
          <strong>Experts Actifs :</strong> <span style={{ color: '#16a34a', fontWeight: 'bold' }}>{activeCount}</span>
        </span>
        <span>
          <strong>Experts Morts :</strong>{' '}
          <span style={{ color: totalDead > 0 ? '#dc2626' : '#16a34a', fontWeight: 'bold' }}>
            {totalDead}
          </span>
        </span>
        <span>
          <strong>Expert #1 Max :</strong> Couche {maxCell.layer}, Expert {maxCell.expert} ({maxCell.val.toFixed(2)}%)
        </span>
      </div>

      {/* Heatmap interactive grid with scroll support */}
      <div
        style={{
          overflowX: 'auto',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          padding: '12px',
          backgroundColor: '#fafafa',
        }}
      >
        <div style={{ display: 'inline-block', minWidth: '100%' }}>
          {/* Top header: Experts (X-Axis) */}
          <div style={{ display: 'flex', marginLeft: '90px', marginBottom: '6px' }}>
            {Array.from({ length: expertsPerLayer }).map((_, expIdx) => (
              <div
                key={`col-${expIdx}`}
                style={{
                  width: expertsPerLayer > 16 ? '28px' : '48px',
                  minWidth: expertsPerLayer > 16 ? '28px' : '48px',
                  marginRight: '3px',
                  textAlign: 'center',
                  fontSize: '11px',
                  fontWeight: 600,
                  color: '#64748b',
                  userSelect: 'none',
                }}
                title={`Expert #${expIdx}`}
              >
                E{expIdx}
              </div>
            ))}
          </div>

          {/* Rows: Layers (Y-Axis) */}
          {matrix.map((row, layerIdx) => (
            <div
              key={`layer-${layerIdx}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                marginBottom: '4px',
              }}
            >
              {/* Layer label */}
              <div
                style={{
                  width: '85px',
                  minWidth: '85px',
                  fontSize: '12px',
                  fontWeight: 600,
                  color: '#334155',
                  paddingRight: '6px',
                  textAlign: 'right',
                  userSelect: 'none',
                }}
              >
                Couche {layerIdx}
              </div>

              {/* Expert cells */}
              <div style={{ display: 'flex' }}>
                {row.map((val, expIdx) => {
                  const { bg, text, isDead } = getCellColor(val, layerIdx);
                  const isHovered =
                    hoveredCell?.layer === layerIdx && hoveredCell?.expert === expIdx;
                  const displayVal = normMode === 'layer' ? val : val / numLayers;

                  return (
                    <div
                      key={`cell-${layerIdx}-${expIdx}`}
                      data-testid={`heatmap-cell-${layerIdx}-${expIdx}`}
                      role="gridcell"
                      tabIndex={0}
                      aria-label={`Couche ${layerIdx}, Expert ${expIdx} : ${val.toFixed(2)}% (${isDead ? 'Mort' : 'Actif'})`}
                      title={`Couche #${layerIdx}, Expert #${expIdx} : ${val.toFixed(2)}% (${isDead ? 'Mort' : 'Actif'})`}
                      onMouseEnter={() =>
                        setHoveredCell({
                          layer: layerIdx,
                          expert: expIdx,
                          layerVal: val,
                          globalVal: val / numLayers,
                          isDead,
                        })
                      }
                      onFocus={() =>
                        setHoveredCell({
                          layer: layerIdx,
                          expert: expIdx,
                          layerVal: val,
                          globalVal: val / numLayers,
                          isDead,
                        })
                      }
                      style={{
                        width: expertsPerLayer > 16 ? '28px' : '48px',
                        height: '36px',
                        minWidth: expertsPerLayer > 16 ? '28px' : '48px',
                        marginRight: '3px',
                        backgroundColor: bg,
                        color: text,
                        borderRadius: '4px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: expertsPerLayer > 16 ? '9px' : '11px',
                        fontWeight: 600,
                        cursor: 'pointer',
                        transition: 'transform 0.1s ease, box-shadow 0.1s ease',
                        border: isHovered
                          ? '2px solid #2563eb'
                          : isDead
                          ? '1px dashed #cbd5e1'
                          : '1px solid rgba(0,0,0,0.06)',
                        transform: isHovered ? 'scale(1.1)' : 'scale(1)',
                        zIndex: isHovered ? 5 : 1,
                        position: 'relative',
                        boxShadow: isHovered ? '0 4px 10px rgba(37,99,235,0.3)' : 'none',
                      }}
                    >
                      {showValues && (
                        <span>
                          {isDead
                            ? '0%'
                            : displayVal >= 10
                            ? `${displayVal.toFixed(0)}%`
                            : displayVal >= 1
                            ? `${displayVal.toFixed(1)}%`
                            : `<1%`}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Inspector / Tooltip Bar for Hovered Cell */}
      <div
        data-testid="heatmap-tooltip"
        style={{
          marginTop: '14px',
          padding: '12px 16px',
          backgroundColor: '#f8fafc',
          borderRadius: '8px',
          border: '1px solid #cbd5e1',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#1e293b' }}>
            🔍 Détails Cellule :
          </span>
          <span
            style={{
              padding: '3px 8px',
              backgroundColor: '#e2e8f0',
              borderRadius: '4px',
              fontWeight: 600,
              fontSize: '12px',
              color: '#334155',
            }}
          >
            Couche #{activeHover.layer}
          </span>
          <span
            style={{
              padding: '3px 8px',
              backgroundColor: '#e2e8f0',
              borderRadius: '4px',
              fontWeight: 600,
              fontSize: '12px',
              color: '#334155',
            }}
          >
            Expert #{activeHover.expert}
          </span>
          <span
            style={{
              padding: '3px 8px',
              borderRadius: '4px',
              fontSize: '12px',
              fontWeight: 700,
              backgroundColor: activeHover.isDead ? '#fee2e2' : '#dcfce7',
              color: activeHover.isDead ? '#dc2626' : '#15803d',
            }}
          >
            {activeHover.isDead ? '💀 Expert Mort (0%)' : '⚡ Expert Actif'}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: '#1e293b' }}>
          <span>
            <strong>Part couche :</strong>{' '}
            <span style={{ color: '#2563eb', fontWeight: 600 }}>
              {activeHover.layerVal.toFixed(2)}%
            </span>
          </span>
          <span>
            <strong>Part globale :</strong>{' '}
            <span style={{ color: '#7c3aed', fontWeight: 600 }}>
              {activeHover.globalVal.toFixed(2)}%
            </span>
          </span>
        </div>
      </div>

      {/* Legend */}
      <div
        style={{
          marginTop: '12px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '10px',
          fontSize: '11px',
          color: '#64748b',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span>Intensité d'activation :</span>
          <div
            style={{
              width: '120px',
              height: '10px',
              borderRadius: '5px',
              background: 'linear-gradient(to right, rgb(224, 231, 255), rgb(49, 46, 129))',
            }}
          />
          <span>(Faible ➔ Forte)</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <span
              style={{
                width: '12px',
                height: '12px',
                backgroundColor: '#f1f5f9',
                border: '1px dashed #cbd5e1',
                borderRadius: '2px',
                display: 'inline-block',
              }}
            />
            Expert mort (0%)
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <span
              style={{
                width: '12px',
                height: '12px',
                backgroundColor: '#312e81',
                borderRadius: '2px',
                display: 'inline-block',
              }}
            />
            Expert dominant
          </span>
        </div>
      </div>
    </div>
  );
};
