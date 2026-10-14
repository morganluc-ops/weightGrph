import React, { useState } from 'react';
import { DomainAnalysisResult } from '../types';

interface ExpertTableProps {
  analysis: DomainAnalysisResult | null;
  loading: boolean;
}

export const ExpertTable: React.FC<ExpertTableProps> = ({ analysis, loading }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!analysis) return;
    const jsonStr = JSON.stringify(analysis, null, 2);
    const textToCopy = `Voici l'analyse des activations des experts MoE :\n${jsonStr}`;
    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch((err) => {
      console.error('Failed to copy: ', err);
    });
  };

  if (loading) {
    return <div data-testid="loading-spinner" style={{ padding: '20px' }}>Analyse des poids MoE en cours...</div>;
  }

  if (!analysis || !analysis.experts || analysis.experts.length === 0) {
    return <div style={{ padding: '20px' }}>Aucune donnée disponible pour ce domaine.</div>;
  }

  return (
    <div style={{ marginTop: '20px', overflowX: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div>
          <h3 style={{ margin: 0 }}>
            Répartition de l'utilisation des experts / poids pour : <span style={{ color: '#0066cc' }}>{analysis.domain_name}</span>
          </h3>
          <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>
            L'utilisation (%) correspond à la part relative de masse de routage (somme des probabilités softmax de l'expert divisée par la masse totale du modèle).
          </div>
        </div>
        <button
          onClick={handleCopy}
          style={{
            padding: '6px 12px',
            backgroundColor: copied ? '#27ae60' : '#2980b9',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '14px',
            transition: 'background-color 0.2s',
          }}
        >
          {copied ? '✅ Copié !' : '📋 Copy Analysis'}
        </button>
      </div>
      <table
        aria-label="Experts Table"
        style={{
          width: '100%',
          borderCollapse: 'collapse',
          boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
          fontFamily: 'sans-serif',
        }}
      >
        <thead>
          <tr style={{ backgroundColor: '#2c3e50', color: '#ffffff', textAlign: 'left' }}>
            <th style={{ padding: '12px 15px' }}>Rang</th>
            <th style={{ padding: '12px 15px' }}>Couche (Layer)</th>
            <th style={{ padding: '12px 15px' }}>ID Expert</th>
            <th style={{ padding: '12px 15px' }}>Nom / Spécialité de l'Expert</th>
            <th style={{ padding: '12px 15px' }}>Taille (Poids)</th>
            <th
              style={{ padding: '12px 15px', cursor: 'help' }}
              title="Calcul de l'utilisation : Somme des probabilités de routage (softmax gating) de l'expert sur tous les tokens / Total des probabilités de routage de tous les experts * 100"
            >
              Utilisation (%) ℹ️
            </th>
          </tr>
        </thead>
        <tbody>
          {analysis.experts.map((expert, index) => (
            <tr
              key={`${expert.layer_id}-${expert.expert_id}`}
              style={{
                backgroundColor: index % 2 === 0 ? '#f8f9fa' : '#ffffff',
                borderBottom: '1px solid #dddddd',
              }}
            >
              <td style={{ padding: '12px 15px', fontWeight: 'bold' }}>#{index + 1}</td>
              <td style={{ padding: '12px 15px' }}>Couche {expert.layer_id}</td>
              <td style={{ padding: '12px 15px' }}>Expert {expert.expert_id}</td>
              <td style={{ padding: '12px 15px' }}>{expert.expert_name}</td>
              <td style={{ padding: '12px 15px' }}>{expert.num_params}</td>
              <td style={{ padding: '12px 15px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div
                    style={{
                      width: '100px',
                      height: '14px',
                      backgroundColor: '#e0e0e0',
                      borderRadius: '7px',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      data-testid={`bar-${index}`}
                      style={{
                        width: `${Math.min(100, Math.max(0, expert.usage_percentage))}%`,
                        height: '100%',
                        backgroundColor: index === 0 ? '#27ae60' : '#2980b9',
                        transition: 'width 0.3s ease-in-out',
                      }}
                    />
                  </div>
                  <span style={{ fontWeight: 'bold', minWidth: '50px' }}>
                    {expert.usage_percentage.toFixed(1)} %
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
