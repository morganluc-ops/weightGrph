import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PromptAnalyzer } from './PromptAnalyzer';
import * as api from '../api';

vi.mock('../api', () => ({
  fetchPrompts: vi.fn(),
  uploadPromptFile: vi.fn(),
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

  it('renders dropdown with prompts and does not render free-text input', async () => {
    render(
      <PromptAnalyzer
        onAnalyzePrompt={vi.fn()}
        loading={false}
        domainsMap={mockDomains}
      />
    );

    expect(screen.queryByTestId('custom-prompt-input')).not.toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
      expect(screen.getByRole('option', { name: /philosophie/i })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: /code/i })).toBeInTheDocument();
    });
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
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'code.md' } });

    const analyzeBtn = screen.getByRole('button', { name: /Analyser le Prompt/i });
    expect(analyzeBtn).not.toBeDisabled();
    fireEvent.click(analyzeBtn);

    expect(handleAnalyze).toHaveBeenCalledWith('code.md');
  });

  it('uploads a .md file, adds it to the dropdown, and selects it', async () => {
    const handleAnalyze = vi.fn();
    (api.uploadPromptFile as any).mockResolvedValue({
      id: 'nouveau_prompt.md',
      name: 'nouveau_prompt',
    });

    render(
      <PromptAnalyzer
        onAnalyzePrompt={handleAnalyze}
        loading={false}
        domainsMap={mockDomains}
      />
    );

    const fileInput = screen.getByTestId('upload-prompt-input');
    const file = new File(['# Analyse IA'], 'nouveau_prompt.md', { type: 'text/markdown' });

    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(api.uploadPromptFile).toHaveBeenCalledWith(file);
      expect(screen.getByRole('option', { name: /nouveau_prompt/i })).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox') as HTMLSelectElement;
    expect(select.value).toBe('nouveau_prompt.md');

    fireEvent.click(screen.getByRole('button', { name: /Analyser le Prompt/i }));
    expect(handleAnalyze).toHaveBeenCalledWith('nouveau_prompt.md');
  });

  it('shows an error when uploading a non-.md file', async () => {
    render(
      <PromptAnalyzer
        onAnalyzePrompt={vi.fn()}
        loading={false}
        domainsMap={mockDomains}
      />
    );

    const fileInput = screen.getByTestId('upload-prompt-input');
    const invalidFile = new File(['not markdown'], 'script.py', { type: 'text/x-python' });

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    expect(api.uploadPromptFile).not.toHaveBeenCalled();
    expect(screen.getByText(/Seuls les fichiers \.md sont autorisés/i)).toBeInTheDocument();
  });

  it('accepts files with uppercase .MD extension', async () => {
    (api.uploadPromptFile as any).mockResolvedValue({
      id: 'UPPER.md',
      name: 'UPPER',
    });

    render(
      <PromptAnalyzer
        onAnalyzePrompt={vi.fn()}
        loading={false}
        domainsMap={mockDomains}
      />
    );

    const fileInput = screen.getByTestId('upload-prompt-input');
    const file = new File(['# Content'], 'UPPER.MD', { type: 'text/markdown' });

    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(api.uploadPromptFile).toHaveBeenCalledWith(file);
    });
  });

  it('displays API error message on upload conflict (409) or size limit (413)', async () => {
    (api.uploadPromptFile as any).mockRejectedValue(
      new Error("Un prompt nommé 'code.md' existe déjà")
    );

    render(
      <PromptAnalyzer
        onAnalyzePrompt={vi.fn()}
        loading={false}
        domainsMap={mockDomains}
      />
    );

    const fileInput = screen.getByTestId('upload-prompt-input');
    const file = new File(['# Code duplicate'], 'code.md', { type: 'text/markdown' });

    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => {
      const errorEl = screen.getByTestId('upload-error');
      expect(errorEl).toHaveTextContent("Un prompt nommé 'code.md' existe déjà");
    });
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
