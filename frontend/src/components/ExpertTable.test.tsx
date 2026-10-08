import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ExpertTable } from './ExpertTable';
import { DomainAnalysisResult } from '../types';

describe('ExpertTable Component', () => {
  const mockAnalysis: DomainAnalysisResult = {
    domain_id: 'philosophie',
    domain_name: 'Philosophie & Éthique',
    experts: [
      {
        layer_id: 0,
        expert_id: 3,
        usage_percentage: 37.4,
        num_params: '1.5B',
        expert_name: 'MoE Layer 0 - Expert 3 (Conceptuel)',
      },
      {
        layer_id: 1,
        expert_id: 1,
        usage_percentage: 28.2,
        num_params: '1.5B',
        expert_name: 'MoE Layer 1 - Expert 1 (Éthique)',
      },
    ],
  };

  it('renders loading state when loading is true', () => {
    render(<ExpertTable analysis={null} loading={true} />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    expect(screen.getByText(/Analyse des poids MoE en cours/i)).toBeInTheDocument();
  });

  it('renders table with sorted expert data', () => {
    render(<ExpertTable analysis={mockAnalysis} loading={false} />);

    expect(screen.getByText(/Philosophie & Éthique/i)).toBeInTheDocument();
    expect(screen.getByText('MoE Layer 0 - Expert 3 (Conceptuel)')).toBeInTheDocument();
    expect(screen.getByText('37.4 %')).toBeInTheDocument();
    expect(screen.getByText('MoE Layer 1 - Expert 1 (Éthique)')).toBeInTheDocument();
    expect(screen.getByText('28.2 %')).toBeInTheDocument();
  });

  it('displays correct rank order #1 and #2', () => {
    render(<ExpertTable analysis={mockAnalysis} loading={false} />);

    expect(screen.getByText('#1')).toBeInTheDocument();
    expect(screen.getByText('#2')).toBeInTheDocument();
  });
});
