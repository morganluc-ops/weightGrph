import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PromptAnalyzer } from './PromptAnalyzer';
import * as api from '../api';

vi.mock('../api', () => ({
  fetchPrompts: vi.fn(),
}));

describe('PromptAnalyzer Component', () => {
  const mockDomains = {
    code: 'Programmation & Génie Logiciel',
    mathematiques: 'Mathématiques & Logique',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (api.fetchPrompts as any).mockResolvedValue([
      { id: 'philosophie.md', name: 'philosophie' },
      { id: 'code.md', name: 'code' },
    ]);
  });

  it('submits free-text prompt when typed into input field', async () => {
    const handleAnalyze = vi.fn();
    render(
      <PromptAnalyzer
        onAnalyzePrompt={handleAnalyze}
        loading={false}
        domainsMap={mockDomains}
      />
    );

    const input = screen.getByTestId('custom-prompt-input');
    fireEvent.change(input, { target: { value: 'Algorithme de tri fusion en Python' } });
    fireEvent.click(screen.getByRole('button', { name: /Analyser le Prompt/i }));

    expect(handleAnalyze).toHaveBeenCalledWith('Algorithme de tri fusion en Python');
  });

  it('selects and submits predefined .md prompt from dropdown', async () => {
    const handleAnalyze = vi.fn();
    render(
      <PromptAnalyzer
        onAnalyzePrompt={handleAnalyze}
        loading={false}
        domainsMap={mockDomains}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'code' })).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'code.md' } });
    fireEvent.click(screen.getByRole('button', { name: /Analyser le Prompt/i }));

    expect(handleAnalyze).toHaveBeenCalledWith('code.md');
  });

  it('toggles data quality warning alert when prevention button is clicked', async () => {
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

