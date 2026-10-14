import React, { useState, useEffect } from 'react';
import { PromptComparisonResult, DivergentExpert, DomainsMap } from '../types';
import { comparePrompts, fetchPrompts, PromptOption } from '../api';

interface PromptComparatorProps {
  domains?: DomainsMap;
}

interface HoveredDiffCell {
  layer: number;
  expert: number;
  valA: number;
  valB: number;
  diff: number;
}

export const PromptComparator: React.FC<PromptComparatorProps> = ({ domains = {} }) => {
  const [promptsList, setPromptsList] = useState<PromptOption[]>([]);
  const [promptA, setPromptA] = useState<string>('code.md');
  const [promptB, setPromptB] = useState<string>('philosophie.md');
  const [customPromptA, setCustomPromptA] = useState<string>('');
  const [customPromptB, setCustomPromptB] = useState<string>('');
  const [useCustomA, setUseCustomA] = useState<boolean>(false);
  const [useCustomB, setUseCustomB] = useState<boolean>(false);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [comparisonResult, setComparisonResult] = useState<PromptComparisonResult | null>(null);
  const [showValues, setShowValues] = useState<boolean>(true);
  const [hoveredCell, setHoveredCell] = useState<HoveredDiffCell | null>(null);

  useEffect(() => {
    async function loadPrompts() {
      try {
        const list = await fetchPrompts();
        setPromptsList(list);
        if (list.length >= 2) {
          setPromptA(list[0].id);
          setPromptB(list[1].id);
        } else if (list.length === 1) {
          setPromptA(list[0].id);
        }
      } catch (err) {
        // Fallback silently if prompts files cannot be fetched
      }
    }
    loadPrompts();
  }, []);

  const handleCompare = async () => {
    const finalA = useCustomA ? customPromptA.trim() : promptA;
    const finalB = useCustomB ? customPromptB.trim() : promptB;

    if (!finalA) {
      setError('Veuillez sélectionner ou saisir le Prompt A.');
      return;
    }
    if (!finalB) {
      setError('Veuillez sélectionner ou saisir le Prompt B.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const result = await comparePrompts(finalA, finalB);
      setComparisonResult(result);
    } catch (err: any) {
      setError(err.message || 'Erreur lors de la comparaison des prompts.');
    } finally {
      setLoading(false);
    }
  };

  const handleSwap = () => {
    const tempPrompt = promptA;
    setPromptA(promptB);
    setPromptB(tempPrompt);

    const tempCustom = customPromptA;
    setCustomPromptA(customPromptB);
    setCustomPromptB(tempCustom);

    const tempUseCustom = useCustomA;
    setUseCustomA(useCustomB);
    setUseCustomB(tempUseCustom);
  };

  // Compute maximum absolute difference for color normalization
  const maxAbsDiff = comparisonResult
    ? Math.max(
        0.001,
        ...comparisonResult.diff_matrix.flatMap((row) => row.map((d) => Math.abs(d)))
      )
    : 1.0;

  const getCellColor = (diff: number) => {
    if (diff === 0) {
      return { bg: '#f1f5f9', text: '#64748b' };
    }
    const ratio = Math.min(1.0, Math.max(0.12, Math.abs(diff) / maxAbsDiff));
    if (diff > 0) {
      // Prompt A dominant: Bleu indigo
      const alpha = Math.round(ratio * 255);
      const bg = `rgba(37, 99, 235, ${ratio.toFixed(2)})`;
      const text = ratio > 0.5 ? '#ffffff' : '#1e3a8a';
      return { bg, text };
    } else {
      // Prompt B dominant: Orange / vermillon
      const bg = `rgba(234, 88, 12, ${ratio.toFixed(2)})`;
      const text = ratio > 0.5 ? '#ffffff' : '#7c2d12';
      return { bg, text };
    }
  };

  const getDivergenceBadge = (jsd: number) => {
    if (jsd <= 0.0001) {
      return { label: 'Identique (0.0)', bg: '#dcfce7', text: '#15803d' };
    }
    if (jsd < 0.15) {
      return { label: 'Divergence Faible', bg: '#e0f2fe', text: '#0369a1' };
    }
    if (jsd < 0.4) {
      return { label: 'Divergence Modérée', bg: '#fef3c7', text: '#b45309' };
    }
    return { label: 'Divergence Forte', bg: '#fee2e2', text: '#b91c1c' };
  };

  const domainEntries = Object.entries(domains);

  // Find layer with highest divergence
  let maxDivergentLayer = 0;
  let maxLayerJsd = 0;
  if (comparisonResult && comparisonResult.layer_jsd.length > 0) {
    comparisonResult.layer_jsd.forEach((jsd, idx) => {
      if (jsd > maxLayerJsd) {
        maxLayerJsd = jsd;
        maxDivergentLayer = idx;
      }
    });
  }

  // Count dominance
  let countA = 0;
  let countB = 0;
  if (comparisonResult) {
    comparisonResult.diff_matrix.forEach((row) => {
      row.forEach((d) => {
        if (d > 0) countA++;
        else if (d < 0) countB++;
      });
    });
  }

  return (
    <div
      data-testid="prompt-comparator"
      style={{
        backgroundColor: '#ffffff',
        borderRadius: '10px',
        padding: '24px',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
        marginBottom: '24px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '20px', color: '#1e293b' }}>
            ⚖️ Comparateur de Prompts & Analyse Différentielle
          </h2>
          <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: '#64748b' }}>
            Comparez l'activation couche × expert et mesurez la divergence de Jensen-Shannon (JSD) entre deux requêtes
          </p>
        </div>
      </div>

      {/* Selectors grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr auto 1fr',
          gap: '16px',
          alignItems: 'center',
          backgroundColor: '#f8fafc',
          padding: '18px',
          borderRadius: '8px',
          border: '1px solid #e2e8f0',
          marginBottom: '20px',
        }}
      >
        {/* Prompt A */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <label style={{ fontSize: '14px', fontWeight: 'bold', color: '#2563eb' }}>
              🟦 Prompt A (Bleu)
            </label>
            <button
              type="button"
              onClick={() => setUseCustomA(!useCustomA)}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '12px',
                color: '#2563eb',
                cursor: 'pointer',
                textDecoration: 'underline',
              }}
            >
              {useCustomA ? 'Choisir prédéfini' : 'Texte libre'}
            </button>
          </div>

          {useCustomA ? (
            <textarea
              data-testid="custom-prompt-a-input"
              value={customPromptA}
              onChange={(e) => setCustomPromptA(e.target.value)}
              placeholder="Saisissez le texte du prompt A..."
              rows={3}
              style={{
                width: '100%',
                padding: '8px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '13px',
                resize: 'vertical',
                boxSizing: 'border-box',
              }}
            />
          ) : (
            <select
              data-testid="select-prompt-a"
              value={promptA}
              onChange={(e) => setPromptA(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '14px',
                backgroundColor: '#ffffff',
                cursor: 'pointer',
              }}
            >
              <optgroup label="Fichiers prompts (.md)">
                {promptsList.map((p) => (
                  <option key={`a-file-${p.id}`} value={p.id}>
                    📄 {p.name} ({p.id})
                  </option>
                ))}
              </optgroup>
              {domainEntries.length > 0 && (
                <optgroup label="Domaines prédéfinis">
                  {domainEntries.map(([key, name]) => (
                    <option key={`a-dom-${key}`} value={key}>
                      🌐 {name}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          )}
        </div>

        {/* Swap button */}
        <div style={{ textAlign: 'center', paddingTop: '16px' }}>
          <button
            type="button"
            data-testid="swap-prompts-btn"
            onClick={handleSwap}
            title="Inverser Prompt A et Prompt B"
            style={{
              backgroundColor: '#ffffff',
              border: '1px solid #cbd5e1',
              borderRadius: '50%',
              width: '40px',
              height: '40px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              fontSize: '18px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
              transition: 'background-color 0.2s',
            }}
          >
            ⇄
          </button>
        </div>

        {/* Prompt B */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <label style={{ fontSize: '14px', fontWeight: 'bold', color: '#ea580c' }}>
              🟧 Prompt B (Orange)
            </label>
            <button
              type="button"
              onClick={() => setUseCustomB(!useCustomB)}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '12px',
                color: '#ea580c',
                cursor: 'pointer',
                textDecoration: 'underline',
              }}
            >
              {useCustomB ? 'Choisir prédéfini' : 'Texte libre'}
            </button>
          </div>

          {useCustomB ? (
            <textarea
              data-testid="custom-prompt-b-input"
              value={customPromptB}
              onChange={(e) => setCustomPromptB(e.target.value)}
              placeholder="Saisissez le texte du prompt B..."
              rows={3}
              style={{
                width: '100%',
                padding: '8px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '13px',
                resize: 'vertical',
                boxSizing: 'border-box',
              }}
            />
          ) : (
            <select
              data-testid="select-prompt-b"
              value={promptB}
              onChange={(e) => setPromptB(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '14px',
                backgroundColor: '#ffffff',
                cursor: 'pointer',
              }}
            >
              <optgroup label="Fichiers prompts (.md)">
                {promptsList.map((p) => (
                  <option key={`b-file-${p.id}`} value={p.id}>
                    📄 {p.name} ({p.id})
                  </option>
                ))}
              </optgroup>
              {domainEntries.length > 0 && (
                <optgroup label="Domaines prédéfinis">
                  {domainEntries.map(([key, name]) => (
                    <option key={`b-dom-${key}`} value={key}>
                      🌐 {name}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          )}
        </div>
      </div>

      {/* Action button */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '20px' }}>
        <button
          type="button"
          data-testid="compare-prompts-btn"
          onClick={handleCompare}
          disabled={loading}
          style={{
            backgroundColor: loading ? '#94a3b8' : '#2563eb',
            color: '#ffffff',
            border: 'none',
            borderRadius: '6px',
            padding: '10px 24px',
            fontSize: '15px',
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            boxShadow: '0 2px 4px rgba(37, 99, 235, 0.2)',
            transition: 'background-color 0.2s',
          }}
        >
          {loading ? '⏳ Calcul de la divergence...' : '🔍 Lancer la comparaison'}
        </button>
      </div>

      {error && (
        <div
          data-testid="comparator-error"
          style={{
            backgroundColor: '#fee2e2',
            color: '#b91c1c',
            padding: '12px 16px',
            borderRadius: '6px',
            marginBottom: '20px',
            fontSize: '14px',
          }}
        >
          {error}
        </div>
      )}

      {/* Comparison Results */}
      {comparisonResult && (
        <div data-testid="comparison-results" style={{ marginTop: '24px' }}>
          {/* Summary Metric Cards */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '16px',
              marginBottom: '24px',
            }}
          >
            {/* Global JSD */}
            <div
              style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                padding: '16px',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '13px', color: '#64748b', marginBottom: '6px' }}>
                Divergence de Jensen-Shannon Globale
              </div>
              <div
                data-testid="global-jsd-value"
                style={{ fontSize: '28px', fontWeight: 'bold', color: '#1e293b', marginBottom: '8px' }}
              >
                {comparisonResult.global_jsd.toFixed(4)}
              </div>
              {(() => {
                const badge = getDivergenceBadge(comparisonResult.global_jsd);
                return (
                  <span
                    style={{
                      display: 'inline-block',
                      backgroundColor: badge.bg,
                      color: badge.text,
                      fontSize: '12px',
                      fontWeight: 600,
                      padding: '3px 10px',
                      borderRadius: '12px',
                    }}
                  >
                    {badge.label}
                  </span>
                );
              })()}
            </div>

            {/* Most Divergent Layer */}
            <div
              style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                padding: '16px',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '13px', color: '#64748b', marginBottom: '6px' }}>
                Couche la plus divergente
              </div>
              <div
                data-testid="max-divergent-layer"
                style={{ fontSize: '28px', fontWeight: 'bold', color: '#2563eb', marginBottom: '8px' }}
              >
                Couche {maxDivergentLayer}
              </div>
              <div style={{ fontSize: '12px', color: '#64748b' }}>
                Score JSD couche : <strong>{maxLayerJsd.toFixed(4)}</strong>
              </div>
            </div>

            {/* Dominance summary */}
            <div
              style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                padding: '16px',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '13px', color: '#64748b', marginBottom: '6px' }}>
                Répartition de dominance
              </div>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', marginTop: '10px' }}>
                <div>
                  <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#2563eb' }}>{countA}</span>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>Prompt A dominant</div>
                </div>
                <div style={{ borderRight: '1px solid #cbd5e1' }} />
                <div>
                  <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#ea580c' }}>{countB}</span>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>Prompt B dominant</div>
                </div>
              </div>
            </div>
          </div>

          {/* Per-Layer JSD breakdown */}
          <div
            style={{
              backgroundColor: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '16px',
              marginBottom: '24px',
            }}
          >
            <h3 style={{ margin: '0 0 12px 0', fontSize: '15px', color: '#1e293b' }}>
              📊 Divergence par Couche (JSD locale)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${comparisonResult.num_layers}, 1fr)`, gap: '10px' }}>
              {comparisonResult.layer_jsd.map((jsd, lIdx) => {
                const ratio = Math.min(1.0, maxLayerJsd > 0 ? jsd / maxLayerJsd : 0);
                return (
                  <div
                    key={`layer-jsd-${lIdx}`}
                    style={{
                      backgroundColor: '#ffffff',
                      border: '1px solid #cbd5e1',
                      borderRadius: '6px',
                      padding: '10px',
                      textAlign: 'center',
                    }}
                  >
                    <div style={{ fontSize: '12px', fontWeight: 600, color: '#475569' }}>
                      Couche {lIdx}
                    </div>
                    <div
                      data-testid={`layer-jsd-val-${lIdx}`}
                      style={{ fontSize: '16px', fontWeight: 'bold', color: '#1e293b', margin: '4px 0' }}
                    >
                      {jsd.toFixed(4)}
                    </div>
                    <div style={{ height: '6px', backgroundColor: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.round(ratio * 100)}%`,
                          backgroundColor: ratio > 0.7 ? '#dc2626' : ratio > 0.4 ? '#f59e0b' : '#3b82f6',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Differential Heatmap Section */}
          <div
            style={{
              backgroundColor: '#ffffff',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '20px',
              marginBottom: '24px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div>
                <h3 style={{ margin: 0, fontSize: '16px', color: '#1e293b' }}>
                  🗺️ Heatmap Différentielle : Δ(Activation) = Prompt A - Prompt B
                </h3>
                <span style={{ fontSize: '12px', color: '#64748b' }}>
                  Couleur bleue : Prompt A dominant | Couleur orange : Prompt B dominant
                </span>
              </div>
              <label style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showValues}
                  onChange={(e) => setShowValues(e.target.checked)}
                />
                Afficher valeurs numériques (Δ%)
              </label>
            </div>

            {/* Color Legend Bar */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 16px',
                backgroundColor: '#f8fafc',
                borderRadius: '6px',
                marginBottom: '16px',
                fontSize: '12px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '16px', height: '16px', backgroundColor: '#ea580c', borderRadius: '3px' }} />
                <span>Prompt B dominant (jusqu'à -{maxAbsDiff.toFixed(1)}%)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '16px', height: '16px', backgroundColor: '#f1f5f9', border: '1px solid #cbd5e1', borderRadius: '3px' }} />
                <span>Égalité (0.0%)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '16px', height: '16px', backgroundColor: '#2563eb', borderRadius: '3px' }} />
                <span>Prompt A dominant (jusqu'à +{maxAbsDiff.toFixed(1)}%)</span>
              </div>
            </div>

            {/* Matrix grid */}
            <div style={{ overflowX: 'auto' }}>
              <table
                data-testid="diff-heatmap-table"
                style={{
                  borderCollapse: 'separate',
                  borderSpacing: '4px',
                  width: '100%',
                  textAlign: 'center',
                }}
              >
                <thead>
                  <tr>
                    <th style={{ padding: '6px', fontSize: '12px', color: '#64748b' }}>Couche</th>
                    {Array.from({ length: comparisonResult.experts_per_layer }).map((_, expIdx) => (
                      <th
                        key={`header-exp-${expIdx}`}
                        style={{ padding: '6px', fontSize: '12px', color: '#475569', minWidth: '45px' }}
                      >
                        Exp {expIdx}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {comparisonResult.diff_matrix.map((row, layerIdx) => (
                    <tr key={`diff-row-${layerIdx}`}>
                      <td style={{ fontWeight: 'bold', fontSize: '12px', color: '#334155', padding: '6px' }}>
                        L{layerIdx}
                      </td>
                      {row.map((diffVal, expIdx) => {
                        const { bg, text } = getCellColor(diffVal);
                        const valA = comparisonResult.matrix_a[layerIdx][expIdx];
                        const valB = comparisonResult.matrix_b[layerIdx][expIdx];

                        return (
                          <td
                            key={`cell-${layerIdx}-${expIdx}`}
                            data-testid={`diff-cell-${layerIdx}-${expIdx}`}
                            onMouseEnter={() =>
                              setHoveredCell({
                                layer: layerIdx,
                                expert: expIdx,
                                valA,
                                valB,
                                diff: diffVal,
                              })
                            }
                            onMouseLeave={() => setHoveredCell(null)}
                            style={{
                              backgroundColor: bg,
                              color: text,
                              padding: '8px 4px',
                              borderRadius: '4px',
                              fontSize: '11px',
                              fontWeight: '600',
                              cursor: 'pointer',
                              transition: 'transform 0.1s, box-shadow 0.1s',
                              userSelect: 'none',
                            }}
                          >
                            {showValues ? `${diffVal > 0 ? '+' : ''}${diffVal.toFixed(1)}%` : ''}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Hovered cell info panel */}
            {hoveredCell && (
              <div
                data-testid="hovered-cell-info"
                style={{
                  marginTop: '12px',
                  padding: '10px 16px',
                  backgroundColor: '#f1f5f9',
                  borderRadius: '6px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  fontSize: '13px',
                }}
              >
                <div>
                  <strong>Couche {hoveredCell.layer} - Expert {hoveredCell.expert}</strong> :{' '}
                  Prompt A: <span style={{ color: '#2563eb', fontWeight: 'bold' }}>{hoveredCell.valA.toFixed(2)}%</span>{' '}|{' '}
                  Prompt B: <span style={{ color: '#ea580c', fontWeight: 'bold' }}>{hoveredCell.valB.toFixed(2)}%</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>
                    Écart :{' '}
                    <strong style={{ color: hoveredCell.diff > 0 ? '#2563eb' : hoveredCell.diff < 0 ? '#ea580c' : '#64748b' }}>
                      {hoveredCell.diff > 0 ? `+${hoveredCell.diff.toFixed(2)}%` : `${hoveredCell.diff.toFixed(2)}%`}
                    </strong>
                  </span>
                  <span
                    style={{
                      padding: '2px 8px',
                      borderRadius: '10px',
                      fontSize: '11px',
                      fontWeight: 'bold',
                      backgroundColor:
                        hoveredCell.diff > 0 ? '#dbeafe' : hoveredCell.diff < 0 ? '#ffedd5' : '#e2e8f0',
                      color:
                        hoveredCell.diff > 0 ? '#1d4ed8' : hoveredCell.diff < 0 ? '#c2410c' : '#475569',
                    }}
                  >
                    {hoveredCell.diff > 0 ? 'Prompt A dominant' : hoveredCell.diff < 0 ? 'Prompt B dominant' : 'Neutre'}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Top Divergent Experts Table */}
          <div
            style={{
              backgroundColor: '#ffffff',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '20px',
            }}
          >
            <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', color: '#1e293b' }}>
              🎯 Experts les plus Clivants / Différenciants
            </h3>
            <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: '#64748b' }}>
              Experts dont la charge d'activation varie le plus entre les deux requêtes
            </p>

            <div style={{ overflowX: 'auto' }}>
              <table
                data-testid="divergent-experts-table"
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  textAlign: 'left',
                  fontSize: '13px',
                }}
              >
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#64748b' }}>
                    <th style={{ padding: '8px 12px' }}>Rang</th>
                    <th style={{ padding: '8px 12px' }}>Expert MoE</th>
                    <th style={{ padding: '8px 12px' }}>Activation A</th>
                    <th style={{ padding: '8px 12px' }}>Activation B</th>
                    <th style={{ padding: '8px 12px' }}>Différence (Δ%)</th>
                    <th style={{ padding: '8px 12px' }}>Dominance</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonResult.top_divergent_experts.map((exp: DivergentExpert, idx: number) => {
                    const isADominant = exp.diff > 0;
                    return (
                      <tr
                        key={`div-exp-${exp.layer_id}-${exp.expert_id}`}
                        data-testid={`divergent-expert-row-${idx}`}
                        style={{
                          borderBottom: '1px solid #f1f5f9',
                          backgroundColor: idx % 2 === 0 ? '#ffffff' : '#fafafa',
                        }}
                      >
                        <td style={{ padding: '8px 12px', fontWeight: 'bold', color: '#64748b' }}>
                          #{idx + 1}
                        </td>
                        <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>
                          {exp.expert_name}
                        </td>
                        <td style={{ padding: '8px 12px', color: '#2563eb', fontWeight: 'bold' }}>
                          {exp.activation_a.toFixed(2)}%
                        </td>
                        <td style={{ padding: '8px 12px', color: '#ea580c', fontWeight: 'bold' }}>
                          {exp.activation_b.toFixed(2)}%
                        </td>
                        <td style={{ padding: '8px 12px', fontWeight: 'bold' }}>
                          <span
                            style={{
                              color: isADominant ? '#2563eb' : exp.diff < 0 ? '#ea580c' : '#64748b',
                            }}
                          >
                            {isADominant ? `+${exp.diff.toFixed(2)}%` : `${exp.diff.toFixed(2)}%`}
                          </span>
                        </td>
                        <td style={{ padding: '8px 12px' }}>
                          <span
                            style={{
                              display: 'inline-block',
                              padding: '2px 8px',
                              borderRadius: '12px',
                              fontSize: '11px',
                              fontWeight: 'bold',
                              backgroundColor: isADominant ? '#dbeafe' : exp.diff < 0 ? '#ffedd5' : '#f1f5f9',
                              color: isADominant ? '#1d4ed8' : exp.diff < 0 ? '#c2410c' : '#64748b',
                            }}
                          >
                            {isADominant ? 'Prompt A' : exp.diff < 0 ? 'Prompt B' : 'Égalité'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
