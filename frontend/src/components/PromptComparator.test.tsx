import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PromptComparator } from './PromptComparator';
import * as api from '../api';
import { PromptComparisonResult } from '../types';

describe('PromptComparator Component', () => {
  const mockDomains = {
    philosophie: 'Philosophie & Éthique',
    code: 'Programmation & Génie Logiciel',
  };

  const mockPrompts = [
    { id: 'code.md', name: 'code' },
    { id: 'philosophie.md', name: 'philosophie' },
  ];

  const mockComparisonResult: PromptComparisonResult = {
    prompt_a_label: 'code.md',
    prompt_b_label: 'philosophie.md',
    matrix_a: [
      [10.0, 90.0],
      [40.0, 60.0],
    ],
    matrix_b: [
      [70.0, 30.0],
      [20.0, 80.0],
    ],
    diff_matrix: [
      [-60.0, 60.0],
      [20.0, -20.0],
    ],
    global_jsd: 0.3245,
    layer_jsd: [0.4521, 0.1234],
    top_divergent_experts: [
      {
        layer_id: 0,
        expert_id: 1,
        expert_name: 'Layer 0 - Expert 1',
        activation_a: 90.0,
        activation_b: 30.0,
        diff: 60.0,
        abs_diff: 60.0,
        dominant_prompt: 'A',
      },
      {
        layer_id: 0,
        expert_id: 0,
        expert_name: 'Layer 0 - Expert 0',
        activation_a: 10.0,
        activation_b: 70.0,
        diff: -60.0,
        abs_diff: 60.0,
        dominant_prompt: 'B',
      },
    ],
    num_layers: 2,
    experts_per_layer: 2,
    model_id: 'test-moe-model',
  };

  beforeEach(() => {
    vi.resetAllMocks();
    vi.spyOn(api, 'fetchPrompts').mockResolvedValue(mockPrompts);
  });

  it('renders title, selectors and compare button', async () => {
    render(<PromptComparator domains={mockDomains} />);

    expect(screen.getByTestId('prompt-comparator')).toBeInTheDocument();
    expect(screen.getByText(/Comparateur de Prompts/i)).toBeInTheDocument();
    expect(screen.getByTestId('select-prompt-a')).toBeInTheDocument();
    expect(screen.getByTestId('select-prompt-b')).toBeInTheDocument();
    expect(screen.getByTestId('compare-prompts-btn')).toBeInTheDocument();
  });

  it('swaps prompt A and prompt B when swap button is clicked', async () => {
    render(<PromptComparator domains={mockDomains} />);

    await waitFor(() => {
      expect(screen.getByTestId('select-prompt-a')).toHaveValue('code.md');
      expect(screen.getByTestId('select-prompt-b')).toHaveValue('philosophie.md');
    });

    const swapBtn = screen.getByTestId('swap-prompts-btn');
    fireEvent.click(swapBtn);

    expect(screen.getByTestId('select-prompt-a')).toHaveValue('philosophie.md');
    expect(screen.getByTestId('select-prompt-b')).toHaveValue('code.md');
  });

  it('allows switching between select dropdown and custom free-text input', async () => {
    render(<PromptComparator domains={mockDomains} />);

    const toggleTextButtons = screen.getAllByText('Texte libre');
    fireEvent.click(toggleTextButtons[0]);

    expect(screen.getByTestId('custom-prompt-a-input')).toBeInTheDocument();
    expect(screen.queryByTestId('select-prompt-a')).not.toBeInTheDocument();

    const togglePreset = screen.getByText('Choisir prédéfini');
    fireEvent.click(togglePreset);

    expect(screen.getByTestId('select-prompt-a')).toBeInTheDocument();
  });

  it('runs comparison and renders JSD scores, differential heatmap, and divergent experts', async () => {
    const compareSpy = vi.spyOn(api, 'comparePrompts').mockResolvedValue(mockComparisonResult);

    render(<PromptComparator domains={mockDomains} />);

    const compareBtn = screen.getByTestId('compare-prompts-btn');
    fireEvent.click(compareBtn);

    await waitFor(() => {
      expect(screen.getByTestId('comparison-results')).toBeInTheDocument();
    });

    expect(compareSpy).toHaveBeenCalledWith('code.md', 'philosophie.md');

    // Check JSD metric cards
    expect(screen.getByTestId('global-jsd-value')).toHaveTextContent('0.3245');
    expect(screen.getByText('Divergence Modérée')).toBeInTheDocument();
    expect(screen.getByTestId('max-divergent-layer')).toHaveTextContent('Couche 0');

    // Check per-layer JSD
    expect(screen.getByTestId('layer-jsd-val-0')).toHaveTextContent('0.4521');
    expect(screen.getByTestId('layer-jsd-val-1')).toHaveTextContent('0.1234');

    // Check differential heatmap table
    expect(screen.getByTestId('diff-heatmap-table')).toBeInTheDocument();
    expect(screen.getByTestId('diff-cell-0-1')).toHaveTextContent('+60.0%');
    expect(screen.getByTestId('diff-cell-0-0')).toHaveTextContent('-60.0%');

    // Hover cell
    fireEvent.mouseEnter(screen.getByTestId('diff-cell-0-1'));
    expect(screen.getByTestId('hovered-cell-info')).toBeInTheDocument();
    expect(screen.getByTestId('hovered-cell-info')).toHaveTextContent('Prompt A: 90.00%');
    expect(screen.getByTestId('hovered-cell-info')).toHaveTextContent('Prompt A dominant');

    // Check divergent experts table
    expect(screen.getByTestId('divergent-experts-table')).toBeInTheDocument();
    expect(screen.getByTestId('divergent-expert-row-0')).toHaveTextContent('Layer 0 - Expert 1');
    expect(screen.getByTestId('divergent-expert-row-0')).toHaveTextContent('+60.00%');
  });

  it('displays error message when comparison API fails', async () => {
    vi.spyOn(api, 'comparePrompts').mockRejectedValue(new Error('Erreur interne du serveur'));

    render(<PromptComparator domains={mockDomains} />);

    const compareBtn = screen.getByTestId('compare-prompts-btn');
    fireEvent.click(compareBtn);

    await waitFor(() => {
      expect(screen.getByTestId('comparator-error')).toBeInTheDocument();
    });

    expect(screen.getByTestId('comparator-error')).toHaveTextContent('Erreur interne du serveur');
  });
});
