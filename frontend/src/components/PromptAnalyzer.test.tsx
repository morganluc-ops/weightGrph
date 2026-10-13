import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PromptAnalyzer } from './PromptAnalyzer';

describe('PromptAnalyzer Component', () => {
  const mockDomains = {
    code: 'Programmation & Génie Logiciel',
    mathematiques: 'Mathématiques & Logique',
  };

  it('submits custom prompt when form is submitted', () => {
    const handleAnalyze = vi.fn();
    render(
      <PromptAnalyzer
        onAnalyzePrompt={handleAnalyze}
        loading={false}
        domainsMap={mockDomains}
      />
    );

    const input = screen.getByLabelText(/Analyse sur Prompt Personnalisé/i);
    fireEvent.change(input, { target: { value: 'Algorithme Python de tri' } });
    fireEvent.click(screen.getByRole('button', { name: /Analyser le Prompt/i }));

    expect(handleAnalyze).toHaveBeenCalledWith('Algorithme Python de tri');
  });

  it('toggles data quality warning alert when prevention button is clicked', () => {
    render(
      <PromptAnalyzer
        onAnalyzePrompt={vi.fn()}
        loading={false}
        domainsMap={mockDomains}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /Prévention & Calibrage/i }));
    expect(screen.getByRole('alert')).toHaveTextContent(/Note de prévention/i);
  });
});
