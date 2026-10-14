import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { RoutingMetricsCard } from './RoutingMetricsCard';
import { BalanceMetrics } from '../types';

describe('RoutingMetricsCard Component', () => {
  const mockMetrics: BalanceMetrics = {
    entropy: 0.8421,
    gini_coefficient: 0.2315,
    dead_experts_count: 2,
    total_experts: 32,
    max_expert_load_share: 14.8,
    layers_metrics: [
      {
        layer_id: 0,
        entropy: 0.9123,
        gini_coefficient: 0.1542,
        dead_experts_count: 0,
        total_experts: 8,
        max_expert_load_share: 18.25,
      },
      {
        layer_id: 1,
        entropy: 0.7812,
        gini_coefficient: 0.3124,
        dead_experts_count: 2,
        total_experts: 8,
        max_expert_load_share: 26.5,
      },
    ],
  };

  it('renders null when balanceMetrics is not provided', () => {
    const { container } = render(<RoutingMetricsCard balanceMetrics={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders all 4 main KPI cards with correct formatted values', () => {
    render(<RoutingMetricsCard balanceMetrics={mockMetrics} />);

    expect(screen.getByTestId('routing-metrics-card')).toBeInTheDocument();

    // Entropy KPI
    const entropyKpi = screen.getByTestId('kpi-entropy');
    expect(entropyKpi).toHaveTextContent('0.842');
    expect(entropyKpi).toHaveTextContent('/ 1.000');
    expect(entropyKpi).toHaveTextContent('Distribution diversifiée');

    // Gini KPI
    const giniKpi = screen.getByTestId('kpi-gini');
    expect(giniKpi).toHaveTextContent('0.232');
    expect(giniKpi).toHaveTextContent('Charge équitable');

    // Dead Experts KPI
    const deadKpi = screen.getByTestId('kpi-dead-experts');
    expect(deadKpi).toHaveTextContent('2');
    expect(deadKpi).toHaveTextContent('/ 32');
    expect(deadKpi).toHaveTextContent('6.3% experts inactifs');

    // Max Load KPI
    const maxLoadKpi = screen.getByTestId('kpi-max-load');
    expect(maxLoadKpi).toHaveTextContent('14.8 %');
  });

  it('renders per-layer metrics table with correct values', () => {
    render(<RoutingMetricsCard balanceMetrics={mockMetrics} />);

    expect(screen.getByTestId('layers-metrics-table')).toBeInTheDocument();
    expect(screen.getByText('Couche 0')).toBeInTheDocument();
    expect(screen.getByText('Couche 1')).toBeInTheDocument();

    expect(screen.getByText('0.912')).toBeInTheDocument();
    expect(screen.getByText('0.154')).toBeInTheDocument();
    expect(screen.getByText('0 / 8')).toBeInTheDocument();
    expect(screen.getByText('18.3 %')).toBeInTheDocument();

    expect(screen.getByText('0.781')).toBeInTheDocument();
    expect(screen.getByText('0.312')).toBeInTheDocument();
    expect(screen.getByText('2 / 8')).toBeInTheDocument();
    expect(screen.getByText('26.5 %')).toBeInTheDocument();
  });

  it('toggles metric guide box when guide button is clicked', () => {
    render(<RoutingMetricsCard balanceMetrics={mockMetrics} />);

    expect(screen.queryByTestId('metrics-guide-box')).not.toBeInTheDocument();

    const guideBtn = screen.getByTestId('toggle-guide-btn');
    fireEvent.click(guideBtn);

    expect(screen.getByTestId('metrics-guide-box')).toBeInTheDocument();
    expect(screen.getByText(/Interprétation des métriques d'équilibrage/i)).toBeInTheDocument();

    fireEvent.click(guideBtn);
    expect(screen.queryByTestId('metrics-guide-box')).not.toBeInTheDocument();
  });

  it('toggles layer details table visibility', () => {
    render(<RoutingMetricsCard balanceMetrics={mockMetrics} />);

    expect(screen.getByTestId('layers-metrics-table')).toBeInTheDocument();

    const toggleBtn = screen.getByTestId('toggle-layers-btn');
    fireEvent.click(toggleBtn);

    expect(screen.queryByTestId('layers-metrics-table')).not.toBeInTheDocument();

    fireEvent.click(toggleBtn);
    expect(screen.getByTestId('layers-metrics-table')).toBeInTheDocument();
  });
});
