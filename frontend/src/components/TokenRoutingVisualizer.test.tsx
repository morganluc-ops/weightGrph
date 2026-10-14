import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { TokenRoutingVisualizer } from './TokenRoutingVisualizer';
import { TokenRoutingInfo } from '../types';

describe('TokenRoutingVisualizer Component', () => {
  const mockTokenRouting: TokenRoutingInfo[] = [
    {
      token: 'import',
      layer_id: 0,
      expert_id: 2,
      confidence: 85.5,
      top_k: [
        { expert_id: 2, weight: 85.5 },
        { expert_id: 5, weight: 14.5 },
      ],
      layers: [
        {
          layer_id: 0,
          expert_id: 2,
          confidence: 85.5,
          top_k: [
            { expert_id: 2, weight: 85.5 },
            { expert_id: 5, weight: 14.5 },
          ],
        },
        {
          layer_id: 1,
          expert_id: 7,
          confidence: 92.0,
          top_k: [
            { expert_id: 7, weight: 92.0 },
            { expert_id: 1, weight: 8.0 },
          ],
        },
      ],
    },
    {
      token: 'torch',
      layer_id: 0,
      expert_id: 4,
      confidence: 76.2,
      top_k: [
        { expert_id: 4, weight: 76.2 },
        { expert_id: 3, weight: 23.8 },
      ],
      layers: [
        {
          layer_id: 0,
          expert_id: 4,
          confidence: 76.2,
          top_k: [
            { expert_id: 4, weight: 76.2 },
            { expert_id: 3, weight: 23.8 },
          ],
        },
        {
          layer_id: 1,
          expert_id: 3,
          confidence: 88.4,
          top_k: [
            { expert_id: 3, weight: 88.4 },
            { expert_id: 6, weight: 11.6 },
          ],
        },
      ],
    },
  ];

  it('renders nothing when tokenRouting is empty or undefined', () => {
    const { container } = render(<TokenRoutingVisualizer tokenRouting={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders tokens with dominant expert badges and confidences for layer 0 by default', () => {
    render(<TokenRoutingVisualizer tokenRouting={mockTokenRouting} numLayers={2} />);

    expect(screen.getByText('import')).toBeInTheDocument();
    expect(screen.getByText('torch')).toBeInTheDocument();

    // Layer 0 dominant experts
    expect(screen.getByText('E2')).toBeInTheDocument();
    expect(screen.getByText('85.5%')).toBeInTheDocument();
    expect(screen.getByText('E4')).toBeInTheDocument();
    expect(screen.getByText('76.2%')).toBeInTheDocument();
  });

  it('switches MoE layers via layer badge buttons and updates dominant experts', () => {
    render(<TokenRoutingVisualizer tokenRouting={mockTokenRouting} numLayers={2} />);

    // Initial state on layer 0
    expect(screen.getByText('E2')).toBeInTheDocument();

    // Click badge Couche 1
    const tab1 = screen.getByTestId('layer-tab-1');
    fireEvent.click(tab1);

    // On layer 1: token 'import' has expert 7 (92%), 'torch' has expert 3 (88.4%)
    expect(screen.getByText('E7')).toBeInTheDocument();
    expect(screen.getByText('92%')).toBeInTheDocument();
    expect(screen.getByText('E3')).toBeInTheDocument();
    expect(screen.getByText('88.4%')).toBeInTheDocument();
  });

  it('switches MoE layers via quick tab buttons', () => {
    render(<TokenRoutingVisualizer tokenRouting={mockTokenRouting} numLayers={2} />);

    const tabLayer1 = screen.getByTestId('layer-tab-1');
    fireEvent.click(tabLayer1);

    expect(screen.getByText('E7')).toBeInTheDocument();
    expect(screen.getByText('92%')).toBeInTheDocument();
  });

  it('displays tooltip with Top-K details on token hover and hides on mouse leave', () => {
    render(<TokenRoutingVisualizer tokenRouting={mockTokenRouting} numLayers={2} />);

    expect(screen.queryByTestId('token-tooltip')).not.toBeInTheDocument();

    const tokenItems = screen.getAllByTestId('token-item');
    fireEvent.mouseEnter(tokenItems[0]);

    // Tooltip should be visible
    const tooltip = screen.getByTestId('token-tooltip');
    expect(tooltip).toBeInTheDocument();
    expect(tooltip).toHaveTextContent(/Token #1 : « import »/i);
    expect(tooltip).toHaveTextContent(/Couche 0/i);
    expect(tooltip).toHaveTextContent(/Expert 2/i);
    expect(tooltip).toHaveTextContent(/Top-2 Gating Network/i);
    expect(tooltip).toHaveTextContent(/Expert 5/i);
    expect(tooltip).toHaveTextContent(/14.5%/i);

    // Mouse leave hides tooltip
    fireEvent.mouseLeave(tokenItems[0]);
    expect(screen.queryByTestId('token-tooltip')).not.toBeInTheDocument();
  });

  it('displays explicit truncation alert when prompt is truncated', () => {
    render(
      <TokenRoutingVisualizer
        tokenRouting={mockTokenRouting}
        numLayers={2}
        truncated={true}
        originalTokenCount={180}
        analyzedTokenCount={128}
      />
    );

    const warning = screen.getByTestId('truncation-warning');
    expect(warning).toBeInTheDocument();
    expect(warning).toHaveTextContent(/Prompt tronqué à l'analyse/i);
    expect(warning).toHaveTextContent(/128 premiers tokens sur 180 tokens/i);
  });

  it('does not display truncation alert when truncated is false', () => {
    render(
      <TokenRoutingVisualizer
        tokenRouting={mockTokenRouting}
        numLayers={2}
        truncated={false}
        originalTokenCount={2}
        analyzedTokenCount={2}
      />
    );

    expect(screen.queryByTestId('truncation-warning')).not.toBeInTheDocument();
  });

  it('handles legacy token routing data gracefully without layers field', () => {
    const legacyData: TokenRoutingInfo[] = [
      {
        token: 'legacy',
        layer_id: 0,
        expert_id: 1,
        confidence: 95.0,
      },
    ];

    render(<TokenRoutingVisualizer tokenRouting={legacyData} />);

    expect(screen.getByText('legacy')).toBeInTheDocument();
    expect(screen.getByText('E1')).toBeInTheDocument();
    expect(screen.getByText('95%')).toBeInTheDocument();
  });
});
