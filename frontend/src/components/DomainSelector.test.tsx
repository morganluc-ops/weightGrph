import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DomainSelector } from './DomainSelector';

describe('DomainSelector Component', () => {
  const mockDomains = {
    philosophie: 'Philosophie & Éthique',
    mathematiques: 'Mathématiques & Logique',
  };

  it('renders domain options correctly', () => {
    render(
      <DomainSelector
        domains={mockDomains}
        selectedDomain="philosophie"
        onSelectDomain={vi.fn()}
      />
    );

    expect(screen.getByLabelText(/Choisir un domaine/i)).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Philosophie & Éthique' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Mathématiques & Logique' })).toBeInTheDocument();
  });

  it('calls onSelectDomain when selection changes', () => {
    const handleSelect = vi.fn();
    render(
      <DomainSelector
        domains={mockDomains}
        selectedDomain="philosophie"
        onSelectDomain={handleSelect}
      />
    );

    fireEvent.change(screen.getByRole('combobox'), {
      target: { value: 'mathematiques' },
    });

    expect(handleSelect).toHaveBeenCalledWith('mathematiques');
  });
});
