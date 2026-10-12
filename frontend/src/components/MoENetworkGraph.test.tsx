import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MoENetworkGraph } from './MoENetworkGraph';
import { DomainAnalysisResult } from '../types';

describe('MoENetworkGraph Component', () => {
  const mockAnalysis: DomainAnalysisResult = {
    domain_id: 'code',
    domain_name: 'Programmation & Génie Logiciel',
    experts: [
      {
        layer_id: 1,
        expert_id: 5,
        usage_percentage: 45.0,
        num_params: '1.5B',
        expert_name: 'MoE Layer 1 - Expert 5 (Syntaxe Code)',
      },
      {
        layer_id: 2,
        expert_id: 3,
        usage_percentage: 25.5,
        num_params: '1.5B',
        expert_name: 'MoE Layer 2 - Expert 3 (Optimization)',
      },
    ],
  };

  it('renders 32 expert nodes across 4 layers and highlights active percentages', () => {
    render(<MoENetworkGraph analysis={mockAnalysis} />);
    expect(screen.getByTestId('moe-network-graph')).toBeInTheDocument();
    expect(screen.getByTestId('expert-node-1-5')).toBeInTheDocument();
    expect(screen.getByTestId('expert-node-2-3')).toBeInTheDocument();
    expect(screen.getByText('45.0%')).toBeInTheDocument();
    expect(screen.getByText('25.5%')).toBeInTheDocument();
  });

  it('updates inspected expert banner when clicking an active expert node', () => {
    render(<MoENetworkGraph analysis={mockAnalysis} />);
    fireEvent.click(screen.getByTestId('expert-node-2-3'));
    expect(screen.getByTestId('inspected-expert-banner')).toHaveTextContent(
      'MoE Layer 2 - Expert 3 (Optimization)'
    );
  });
});
