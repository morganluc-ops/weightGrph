import React, { useState } from 'react';
import { DomainAnalysisResult, ExpertUsage } from '../types';

interface MoENetworkGraphProps {
  analysis: DomainAnalysisResult | null;
  numLayers?: number;
  expertsPerLayer?: number;
}

export const MoENetworkGraph: React.FC<MoENetworkGraphProps> = ({
  analysis,
  numLayers = 4,
  expertsPerLayer = 8,
}) => {
  const [selectedNode, setSelectedNode] = useState<ExpertUsage | null>(null);

  if (!analysis || !analysis.experts || analysis.experts.length === 0) {
    return null;
  }

  // Lookup map: "layer-expert" -> { expert, rank }
  const activeMap = new Map<string, { expert: ExpertUsage; rank: number }>();
  analysis.experts.forEach((exp, idx) => {
    activeMap.set(`${exp.layer_id}-${exp.expert_id}`, { expert: exp, rank: idx + 1 });
  });

  const svgWidth = 880;
  const svgHeight = 360;
  const padX = 110;
  const padY = 45;
  const layerSpacing = (svgWidth - padX * 2) / Math.max(1, numLayers - 1);
  const expertSpacing = (svgHeight - padY * 2) / Math.max(1, expertsPerLayer - 1);

  const getCoord = (layerId: number, expertId: number) => ({
    x: padX + layerId * layerSpacing,
    y: padY + expertId * expertSpacing,
  });

  // Build inter-layer co-activation links between active experts
  const links: Array<{
    key: string;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    strength: number;
    isPrimary: boolean;
  }> = [];

  for (let l = 0; l < numLayers - 1; l++) {
    const currentLayerActives = analysis.experts.filter((e) => e.layer_id === l);
    const nextLayerActives = analysis.experts.filter((e) => e.layer_id === l + 1);

    currentLayerActives.forEach((src) => {
      nextLayerActives.forEach((dst) => {
        const p1 = getCoord(src.layer_id, src.expert_id);
        const p2 = getCoord(dst.layer_id, dst.expert_id);
        const avgUsage = (src.usage_percentage + dst.usage_percentage) / 2;
        links.push({
          key: `link-${src.layer_id}-${src.expert_id}-to-${dst.layer_id}-${dst.expert_id}`,
          x1: p1.x,
          y1: p1.y,
          x2: p2.x,
          y2: p2.y,
          strength: avgUsage,
          isPrimary:
            activeMap.get(`${src.layer_id}-${src.expert_id}`)?.rank === 1 ||
            activeMap.get(`${dst.layer_id}-${dst.expert_id}`)?.rank === 1,
        });
      });
    });
  }

  const inspected = selectedNode || analysis.experts[0];

  return (
    <div
      data-testid="moe-network-graph"
      style={{
        marginTop: '24px',
        padding: '18px',
        backgroundColor: '#0f172a',
        borderRadius: '10px',
        color: '#f8fafc',
        boxShadow: '0 4px 14px rgba(15, 23, 42, 0.25)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '12px' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '18px', color: '#f1f5f9' }}>
            🕸️ Topologie du Réseau Neuronal MoE ({numLayers} couches × {expertsPerLayer} experts)
          </h3>
          <span style={{ fontSize: '12px', color: '#94a3b8' }}>
            Cliquez sur un nœud illuminé pour inspecter les poids et le routage Top-K de l'expert
          </span>
        </div>
        <div style={{ display: 'flex', gap: '14px', fontSize: '12px', color: '#cbd5e1' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#10b981', display: 'inline-block' }} />
            Expert Dominant (#1)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#38bdf8', display: 'inline-block' }} />
            Expert Co-activé
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#334155', display: 'inline-block' }} />
            Expert Dormant
          </span>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ width: '100%', height: 'auto', minWidth: '640px', display: 'block' }}
          aria-label="MoE Neural Network Graph"
        >
          <defs>
            <filter id="glow-primary" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Subtle background grid lines for all layers */}
          {Array.from({ length: numLayers }).map((_, layerIdx) => {
            const x = padX + layerIdx * layerSpacing;
            return (
              <g key={`layer-col-${layerIdx}`}>
                <line
                  x1={x}
                  y1={padY - 15}
                  x2={x}
                  y2={svgHeight - 20}
                  stroke="#1e293b"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                />
                <text
                  x={x}
                  y={22}
                  textAnchor="middle"
                  fill="#94a3b8"
                  fontSize="12"
                  fontWeight="bold"
                >
                  Couche {layerIdx}
                </text>
              </g>
            );
          })}

          {/* Co-activation routing links */}
          {links.map((lnk) => (
            <path
              key={lnk.key}
              d={`M ${lnk.x1} ${lnk.y1} C ${(lnk.x1 + lnk.x2) / 2} ${lnk.y1}, ${(lnk.x1 + lnk.x2) / 2} ${lnk.y2}, ${lnk.x2} ${lnk.y2}`}
              fill="none"
              stroke={lnk.isPrimary ? '#10b981' : '#38bdf8'}
              strokeWidth={Math.max(1.5, Math.min(6, lnk.strength / 8))}
              strokeOpacity={lnk.isPrimary ? 0.8 : 0.45}
            />
          ))}

          {/* Expert nodes (4 layers x 8 experts) */}
          {Array.from({ length: numLayers }).map((_, layerId) =>
            Array.from({ length: expertsPerLayer }).map((__, expertId) => {
              const { x, y } = getCoord(layerId, expertId);
              const activeInfo = activeMap.get(`${layerId}-${expertId}`);
              const isActive = Boolean(activeInfo);
              const isRank1 = activeInfo?.rank === 1;
              const isSelected =
                inspected &&
                inspected.layer_id === layerId &&
                inspected.expert_id === expertId;

              const radius = isRank1 ? 14 : isActive ? 11 : 6;
              const fillColor = isRank1
                ? '#10b981'
                : isActive
                ? '#38bdf8'
                : '#334155';

              return (
                <g
                  key={`node-${layerId}-${expertId}`}
                  data-testid={`expert-node-${layerId}-${expertId}`}
                  onClick={() => activeInfo && setSelectedNode(activeInfo.expert)}
                  style={{ cursor: isActive ? 'pointer' : 'default' }}
                >
                  {isSelected && (
                    <circle
                      cx={x}
                      cy={y}
                      r={radius + 5}
                      fill="none"
                      stroke="#facc15"
                      strokeWidth={2}
                    />
                  )}
                  <circle
                    cx={x}
                    cy={y}
                    r={radius}
                    fill={fillColor}
                    filter={isActive ? 'url(#glow-primary)' : undefined}
                    stroke={isActive ? '#0f172a' : 'none'}
                    strokeWidth={1.5}
                  />
                  <text
                    x={x - (isActive ? 20 : 14)}
                    y={y + 4}
                    textAnchor="end"
                    fill={isActive ? '#e2e8f0' : '#64748b'}
                    fontSize="10"
                  >
                    E{expertId}
                  </text>
                  {activeInfo && (
                    <text
                      x={x + radius + 6}
                      y={y + 4}
                      fill={isRank1 ? '#34d399' : '#7dd3fc'}
                      fontSize="11"
                      fontWeight="bold"
                    >
                      {activeInfo.expert.usage_percentage.toFixed(1)}%
                    </text>
                  )}
                </g>
              );
            })
          )}
        </svg>
      </div>

      {inspected && (
        <div
          data-testid="inspected-expert-banner"
          style={{
            marginTop: '12px',
            padding: '10px 14px',
            backgroundColor: '#1e293b',
            borderRadius: '6px',
            borderLeft: '4px solid #10b981',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '8px',
            fontSize: '13px',
          }}
        >
          <div>
            <strong>Expert inspecté :</strong> {inspected.expert_name} (Couche {inspected.layer_id}, ID {inspected.expert_id})
          </div>
          <div style={{ display: 'flex', gap: '16px' }}>
            <span>
              <strong>Poids :</strong> {inspected.num_params}
            </span>
            <span style={{ color: '#34d399', fontWeight: 'bold' }}>
              Activation : {inspected.usage_percentage.toFixed(1)} %
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
