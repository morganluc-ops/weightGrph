import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ExpertTable } from './ExpertTable';

Object.assign(navigator, {
  clipboard: {
    writeText: vi.fn().mockImplementation(() => Promise.resolve()),
  },
});

describe('ExpertTable Component', () => {
  const mockAnalysis = {
    domain_id: 'test_domain',
    domain_name: 'Test Domain',
    is_real_inference: true,
    model_id: 'test-model',
    experts: [
      { layer_id: 0, expert_id: 1, usage_percentage: 85.5, num_params: '1.2M', expert_name: 'Test Expert 1' },
      { layer_id: 1, expert_id: 2, usage_percentage: 14.5, num_params: '1.2M', expert_name: 'Test Expert 2' },
    ],
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders loading spinner when loading is true', () => {
    render(<ExpertTable analysis={null} loading={true} />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('renders no data message when analysis is null', () => {
    render(<ExpertTable analysis={null} loading={false} />);
    expect(screen.getByText('Aucune donnée disponible pour ce domaine.')).toBeInTheDocument();
  });

  it('renders table with experts data', () => {
    render(<ExpertTable analysis={mockAnalysis} loading={false} />);

    expect(screen.getByText(/Répartition de l'utilisation des experts/i)).toBeInTheDocument();
    expect(screen.getByText('Test Domain')).toBeInTheDocument();

    expect(screen.getByText('Test Expert 1')).toBeInTheDocument();
    expect(screen.getByText('85.5 %')).toBeInTheDocument();

    expect(screen.getByText('Test Expert 2')).toBeInTheDocument();
    expect(screen.getByText('14.5 %')).toBeInTheDocument();
  });

  it('copies analysis data to clipboard when button is clicked', async () => {
    render(<ExpertTable analysis={mockAnalysis} loading={false} />);

    const copyButton = screen.getByText('📋 Copy Analysis');
    fireEvent.click(copyButton);

    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
      const calledWith = (navigator.clipboard.writeText as any).mock.calls[0][0];
      expect(calledWith).toContain("Voici l'analyse des activations des experts MoE :");
      expect(calledWith).toContain('test_domain');
      expect(screen.getByText('✅ Copié !')).toBeInTheDocument();
    });
  });
});
