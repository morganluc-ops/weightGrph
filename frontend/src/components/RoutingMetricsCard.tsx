import React, { useState } from 'react';
import { BalanceMetrics } from '../types';

interface RoutingMetricsCardProps {
  balanceMetrics?: BalanceMetrics | null;
}

export const RoutingMetricsCard: React.FC<RoutingMetricsCardProps> = ({ balanceMetrics }) => {
  const [showLayerDetails, setShowLayerDetails] = useState<boolean>(true);
  const [showGuide, setShowGuide] = useState<boolean>(false);

  if (!balanceMetrics) {
    return null;
  }

  const {
    entropy,
    gini_coefficient,
    dead_experts_count,
    total_experts,
    max_expert_load_share,
    layers_metrics,
  } = balanceMetrics;

  const deadPercentage = total_experts > 0 ? (dead_experts_count / total_experts) * 100 : 0;

  // Helpers for KPI badge colors
  const getEntropyColor = (val: number) => {
    if (val >= 0.75) return '#10b981'; // Green
    if (val >= 0.5) return '#f59e0b';  // Amber
    return '#ef4444';                  // Red
  };

  const getGiniColor = (val: number) => {
    if (val <= 0.35) return '#10b981'; // Fair / Low inequality
    if (val <= 0.6) return '#f59e0b';  // Moderate inequality
    return '#ef4444';                  // High inequality
  };

  const getDeadColor = (count: number) => {
    if (count === 0) return '#10b981';
    if (count <= Math.ceil(total_experts * 0.25)) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div
      data-testid="routing-metrics-card"
      style={{
        marginTop: '20px',
        padding: '20px',
        backgroundColor: '#ffffff',
        borderRadius: '10px',
        border: '1px solid #e2e8f0',
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '10px',
          marginBottom: '16px',
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: '18px', color: '#1e293b', display: 'flex', alignItems: 'center', gap: '8px' }}>
            ⚖️ Métriques d'Équilibrage du Routage MoE
          </h3>
          <span style={{ fontSize: '12px', color: '#64748b' }}>
            Indicateurs d'efficacité de routage, dispersion d'entropie et identification des experts dormants
          </span>
        </div>
        <button
          type="button"
          data-testid="toggle-guide-btn"
          onClick={() => setShowGuide(!showGuide)}
          style={{
            padding: '5px 10px',
            backgroundColor: '#f1f5f9',
            border: '1px solid #cbd5e1',
            borderRadius: '6px',
            fontSize: '12px',
            color: '#334155',
            cursor: 'pointer',
            fontWeight: 500,
          }}
        >
          {showGuide ? '✖ Fermer le guide' : 'ℹ️ Guide des métriques'}
        </button>
      </div>

      {/* Guide des métriques explicatif */}
      {showGuide && (
        <div
          data-testid="metrics-guide-box"
          style={{
            backgroundColor: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '12px 16px',
            marginBottom: '16px',
            fontSize: '13px',
            color: '#334155',
            lineHeight: 1.5,
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '6px', color: '#0f172a' }}>
            📖 Interprétation des métriques d'équilibrage :
          </div>
          <ul style={{ margin: '0', paddingLeft: '20px' }}>
            <li>
              <strong>Entropie de routage (Shannon normalisée H / ln(N)) :</strong> Évaluée entre 0 et 1.
              Une valeur proche de 1.0 indique que les experts sont sollicités de manière uniforme. Une valeur proche de 0
              révèle un effondrement du routage (routing collapse) sur un nombre restreint d'experts.
            </li>
            <li>
              <strong>Indice de Gini :</strong> Mesure le degré d'inégalité de la charge entre experts (0 = égalité absolue,
              1 = charge captée à 100% par un unique expert).
            </li>
            <li>
              <strong>Experts morts :</strong> Nombre d'experts n'ayant reçu aucune activation de token lors de l'inférence.
              Un ratio élevé signale une capacité de calcul sous-exploitée.
            </li>
            <li>
              <strong>Charge max expert :</strong> Pourcentage de la masse totale de routage absorbé par l'expert le plus actif.
            </li>
            <li>
              <strong>Calcul de l'utilisation (%) :</strong> Proportion de probabilités de routage (softmax gating) cumulées
              sur tous les tokens analysés.
            </li>
          </ul>
        </div>
      )}

      {/* 4 KPI Grid Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '12px',
          marginBottom: '16px',
        }}
      >
        {/* KPI 1 : Entropie */}
        <div
          data-testid="kpi-entropy"
          title="Entropie de Shannon normalisée H / ln(N) : 1.0 = répartition parfaitement uniforme, 0.0 = goulot sur un seul expert."
          style={{
            backgroundColor: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '14px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
            🌀 Entropie de Routage
          </div>
          <div style={{ margin: '8px 0', display: 'flex', alignItems: 'baseline', gap: '6px' }}>
            <span style={{ fontSize: '24px', fontWeight: 'bold', color: getEntropyColor(entropy) }}>
              {entropy.toFixed(3)}
            </span>
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>/ 1.000</span>
          </div>
          <div>
            <div style={{ width: '100%', height: '6px', backgroundColor: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${Math.min(100, Math.max(0, entropy * 100))}%`,
                  height: '100%',
                  backgroundColor: getEntropyColor(entropy),
                }}
              />
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
              {entropy >= 0.75 ? 'Distribution diversifiée' : entropy >= 0.5 ? 'Dispersion moyenne' : 'Effondrement / biais'}
            </div>
          </div>
        </div>

        {/* KPI 2 : Gini */}
        <div
          data-testid="kpi-gini"
          title="Indice de Gini : 0.0 = parfaite égalité de charge, 1.0 = concentration maximale sur un seul expert."
          style={{
            backgroundColor: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '14px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
            📊 Coefficient de Gini
          </div>
          <div style={{ margin: '8px 0', display: 'flex', alignItems: 'baseline', gap: '6px' }}>
            <span style={{ fontSize: '24px', fontWeight: 'bold', color: getGiniColor(gini_coefficient) }}>
              {gini_coefficient.toFixed(3)}
            </span>
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>[0 - 1]</span>
          </div>
          <div>
            <div style={{ width: '100%', height: '6px', backgroundColor: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${Math.min(100, Math.max(0, gini_coefficient * 100))}%`,
                  height: '100%',
                  backgroundColor: getGiniColor(gini_coefficient),
                }}
              />
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
              {gini_coefficient <= 0.35 ? 'Charge équitable' : gini_coefficient <= 0.6 ? 'Inégalité modérée' : 'Forte inégalité'}
            </div>
          </div>
        </div>

        {/* KPI 3 : Experts Morts */}
        <div
          data-testid="kpi-dead-experts"
          title="Nombre d'experts n'ayant reçu aucune activation lors de cette inférence."
          style={{
            backgroundColor: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '14px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
            💀 Experts Morts (0 Act.)
          </div>
          <div style={{ margin: '8px 0', display: 'flex', alignItems: 'baseline', gap: '6px' }}>
            <span style={{ fontSize: '24px', fontWeight: 'bold', color: getDeadColor(dead_experts_count) }}>
              {dead_experts_count}
            </span>
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>/ {total_experts}</span>
          </div>
          <div>
            <div style={{ width: '100%', height: '6px', backgroundColor: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${Math.min(100, Math.max(0, deadPercentage))}%`,
                  height: '100%',
                  backgroundColor: getDeadColor(dead_experts_count),
                }}
              />
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
              {dead_experts_count === 0 ? 'Tous les experts sont actifs' : `${deadPercentage.toFixed(1)}% experts inactifs`}
            </div>
          </div>
        </div>

        {/* KPI 4 : Charge Max Expert */}
        <div
          data-testid="kpi-max-load"
          title="Part de charge captée par l'expert le plus actif sur l'ensemble du modèle."
          style={{
            backgroundColor: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '14px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
            ⚡ Charge Max Expert
          </div>
          <div style={{ margin: '8px 0', display: 'flex', alignItems: 'baseline', gap: '6px' }}>
            <span style={{ fontSize: '24px', fontWeight: 'bold', color: '#2563eb' }}>
              {max_expert_load_share.toFixed(1)} %
            </span>
          </div>
          <div>
            <div style={{ width: '100%', height: '6px', backgroundColor: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${Math.min(100, Math.max(0, max_expert_load_share))}%`,
                  height: '100%',
                  backgroundColor: '#2563eb',
                }}
              />
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
              Pic unitaire de charge globale
            </div>
          </div>
        </div>
      </div>

      {/* Per-layer Breakdown */}
      {layers_metrics && layers_metrics.length > 0 && (
        <div style={{ marginTop: '14px' }}>
          <button
            type="button"
            data-testid="toggle-layers-btn"
            onClick={() => setShowLayerDetails(!showLayerDetails)}
            style={{
              padding: '6px 12px',
              backgroundColor: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '6px',
              fontSize: '12px',
              fontWeight: 600,
              color: '#334155',
              cursor: 'pointer',
              marginBottom: '10px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            {showLayerDetails ? '▼ Masquer le détail par couche' : '▶ Afficher le détail par couche'} ({layers_metrics.length} couches MoE)
          </button>

          {showLayerDetails && (
            <div style={{ overflowX: 'auto', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
              <table
                data-testid="layers-metrics-table"
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: '12px',
                  fontFamily: 'sans-serif',
                }}
              >
                <thead>
                  <tr style={{ backgroundColor: '#f1f5f9', color: '#334155', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Couche MoE</th>
                    <th style={{ padding: '8px 12px' }}>Entropie (0-1)</th>
                    <th style={{ padding: '8px 12px' }}>Indice de Gini</th>
                    <th style={{ padding: '8px 12px' }}>Experts Morts</th>
                    <th style={{ padding: '8px 12px' }}>Charge Max Couche (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {layers_metrics.map((lm) => (
                    <tr
                      key={`layer-metric-${lm.layer_id}`}
                      style={{ borderBottom: '1px solid #f1f5f9', backgroundColor: '#ffffff' }}
                    >
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>
                        Couche {lm.layer_id}
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontWeight: 600, color: getEntropyColor(lm.entropy), minWidth: '40px' }}>
                            {lm.entropy.toFixed(3)}
                          </span>
                          <div style={{ width: '60px', height: '6px', backgroundColor: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                            <div
                              style={{
                                width: `${Math.min(100, Math.max(0, lm.entropy * 100))}%`,
                                height: '100%',
                                backgroundColor: getEntropyColor(lm.entropy),
                              }}
                            />
                          </div>
                        </div>
                      </td>
                      <td style={{ padding: '8px 12px', color: getGiniColor(lm.gini_coefficient), fontWeight: 600 }}>
                        {lm.gini_coefficient.toFixed(3)}
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <span
                          style={{
                            padding: '2px 8px',
                            borderRadius: '12px',
                            fontSize: '11px',
                            fontWeight: 600,
                            backgroundColor: lm.dead_experts_count === 0 ? '#dcfce7' : '#fee2e2',
                            color: lm.dead_experts_count === 0 ? '#15803d' : '#b91c1c',
                          }}
                        >
                          {lm.dead_experts_count} / {lm.total_experts}
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: '#2563eb' }}>
                        {lm.max_expert_load_share.toFixed(1)} %
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
