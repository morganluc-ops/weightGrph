import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { LayerExpertHeatmap } from './LayerExpertHeatmap';
import { DomainAnalysisResult } from '../types';

describe('LayerExpertHeatmap Component', () => {
  const mockAnalysis: DomainAnalysisResult = {
    domain_id: 'code',
    domain_name: 'Programmation',
    experts: [
      {
        layer_id: 0,
        expert_id: 1,
        usage_percentage: 40.0,
        num_params: '1.2M',
        expert_name: 'Expert #1',
      },
    ],
    layer_expert_matrix: [
      [10.0, 70.0, 0.0, 20.0],
      [0.0, 50.0, 50.0, 0.0],
    ],
  };

  it('renders null when analysis is not provided or matrix is empty', () => {
    const { container: c1 } = render(<LayerExpertHeatmap analysis={null} />);
    expect(c1.firstChild).toBeNull();

    const emptyAnalysis: DomainAnalysisResult = {
      domain_id: 'test',
      domain_name: 'Test',
      experts: [],
      layer_expert_matrix: [],
    };
    const { container: c2 } = render(<LayerExpertHeatmap analysis={emptyAnalysis} />);
    expect(c2.firstChild).toBeNull();
  });

  it('renders the heatmap card, header, and grid cells with correct dimensions', () => {
    render(<LayerExpertHeatmap analysis={mockAnalysis} />);

    expect(screen.getByTestId('layer-expert-heatmap')).toBeInTheDocument();
    expect(
      screen.getByText(/Heatmap Couche × Expert du Routage MoE/i)
    ).toBeInTheDocument();

    // 2 layers x 4 experts = 8 cells
    for (let l = 0; l < 2; l++) {
      for (let e = 0; e < 4; e++) {
        const cell = screen.getByTestId(`heatmap-cell-${l}-${e}`);
        expect(cell).toBeInTheDocument();
        expect(cell).toHaveAttribute('role', 'gridcell');
      }
    }
  });

  it('correctly labels active and dead experts in cells and aria-labels', () => {
    render(<LayerExpertHeatmap analysis={mockAnalysis} />);

    // Cell 0,2 is dead (0%)
    const deadCell = screen.getByTestId('heatmap-cell-0-2');
    expect(deadCell).toHaveAttribute(
      'aria-label',
      expect.stringContaining('Mort')
    );
    expect(deadCell).toHaveTextContent('0%');

    // Cell 0,1 is active (70%)
    const activeCell = screen.getByTestId('heatmap-cell-0-1');
    expect(activeCell).toHaveAttribute(
      'aria-label',
      expect.stringContaining('Actif')
    );
    expect(activeCell).toHaveTextContent('70%');
  });

  it('updates the tooltip / details bar on cell hover with layer, expert, percentage and status', () => {
    render(<LayerExpertHeatmap analysis={mockAnalysis} />);

    const tooltip = screen.getByTestId('heatmap-tooltip');
    expect(tooltip).toBeInTheDocument();

    // Hover on cell (layer 0, expert 1 - active 70%)
    const activeCell = screen.getByTestId('heatmap-cell-0-1');
    fireEvent.mouseEnter(activeCell);

    expect(tooltip).toHaveTextContent('Couche #0');
    expect(tooltip).toHaveTextContent('Expert #1');
    expect(tooltip).toHaveTextContent('70.00%');
    expect(tooltip).toHaveTextContent(/Expert Actif/i);

    // Hover on cell (layer 1, expert 0 - dead 0%)
    const deadCell = screen.getByTestId('heatmap-cell-1-0');
    fireEvent.mouseEnter(deadCell);

    expect(tooltip).toHaveTextContent('Couche #1');
    expect(tooltip).toHaveTextContent('Expert #0');
    expect(tooltip).toHaveTextContent('0.00%');
    expect(tooltip).toHaveTextContent(/Expert Mort/i);
  });

  it('toggles normalization mode between layer and global', () => {
    render(<LayerExpertHeatmap analysis={mockAnalysis} />);

    const normLayerBtn = screen.getByTestId('btn-norm-layer');
    const normGlobalBtn = screen.getByTestId('btn-norm-global');

    // Default is layer normalization
    expect(normLayerBtn).toBeInTheDocument();
    expect(normGlobalBtn).toBeInTheDocument();

    // Switch to global normalization
    fireEvent.click(normGlobalBtn);

    // Cell (0, 1) has layerVal 70%. In global mode with 2 layers, displayed val is 70 / 2 = 35%
    const cell01 = screen.getByTestId('heatmap-cell-0-1');
    expect(cell01).toHaveTextContent('35%');

    // Switch back to layer normalization
    fireEvent.click(normLayerBtn);
    expect(cell01).toHaveTextContent('70%');
  });

  it('toggles numeric values visibility in cells', () => {
    render(<LayerExpertHeatmap analysis={mockAnalysis} />);

    const toggleValuesBtn = screen.getByTestId('btn-toggle-values');
    expect(toggleValuesBtn).toHaveTextContent(/Masquer valeurs/i);

    // Initially values are displayed
    const cell01 = screen.getByTestId('heatmap-cell-0-1');
    expect(cell01).toHaveTextContent('70%');

    // Click to hide values
    fireEvent.click(toggleValuesBtn);
    expect(toggleValuesBtn).toHaveTextContent(/Afficher valeurs/i);
    expect(cell01).toHaveTextContent('');

    // Click again to show values
    fireEvent.click(toggleValuesBtn);
    expect(toggleValuesBtn).toHaveTextContent(/Masquer valeurs/i);
    expect(cell01).toHaveTextContent('70%');
  });
});
