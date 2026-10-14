import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchDomains, fetchDomainAnalysis, analyzeCustomPrompt } from './api';

globalThis.fetch = vi.fn() as any;

describe('Frontend API Client', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('fetchDomains returns available domains', async () => {
    const mockDomains = { philosophie: 'Philosophie & Éthique', code: 'Code' };
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockDomains,
    });

    const domains = await fetchDomains();
    expect(domains).toEqual(mockDomains);
  });

  it('fetchDomainAnalysis returns domain result sorted descending', async () => {
    const mockResult = {
      domain_id: 'philosophie',
      domain_name: 'Philosophie & Éthique',
      experts: [
        { layer_id: 1, expert_id: 0, usage_percentage: 10, num_params: '1.5B', expert_name: 'Exp 0' },
        { layer_id: 0, expert_id: 3, usage_percentage: 37, num_params: '1.5B', expert_name: 'Exp 3' },
      ],
    };

    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });

    const result = await fetchDomainAnalysis('philosophie');
    expect(result.experts[0].usage_percentage).toBe(37);
    expect(result.experts[1].usage_percentage).toBe(10);
  });

  it('analyzeCustomPrompt posts prompt and returns sorted experts', async () => {
    const mockResult = {
      domain_id: 'custom_prompt',
      domain_name: 'Prompt personnalisé (Dominante : Programmation & Génie Logiciel)',
      experts: [
        { layer_id: 3, expert_id: 2, usage_percentage: 25, num_params: '1.5B', expert_name: 'Algorithmes' },
        { layer_id: 1, expert_id: 5, usage_percentage: 45, num_params: '1.5B', expert_name: 'Syntaxe Code' },
      ],
      detected_domains: { code: 100 },
    };

    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });

    const result = await analyzeCustomPrompt('Écris du code Python');
    expect(result.domain_id).toBe('custom_prompt');
    expect(result.experts[0].usage_percentage).toBe(45);
  });

  it('uploadPromptFile sends formData and returns prompt option', async () => {
    const mockFile = new File(['# Prompt content'], 'test.md', { type: 'text/markdown' });
    const mockResponse = { id: 'test.md', name: 'test' };

    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    const { uploadPromptFile } = await import('./api');
    const result = await uploadPromptFile(mockFile);
    expect(result).toEqual(mockResponse);
    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/prompts/upload',
      expect.objectContaining({
        method: 'POST',
        body: expect.any(FormData),
      })
    );
  });

  it('loadMoEModel propagates backend error detail', async () => {
    (fetch as any).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: 'Identifiant de modèle invalide : bad-id' }),
    });

    const { loadMoEModel } = await import('./api');
    await expect(loadMoEModel('bad-id')).rejects.toThrow('Identifiant de modèle invalide : bad-id');
  });

  it('fetchModelProgress calls progress endpoint and returns status', async () => {
    const mockProgress = {
      status: 'downloading',
      model_id: 'test/model',
      progress_percent: 42.5,
      downloaded_bytes: 42500,
      total_bytes: 100000,
      downloaded_mb: 4.2,
      total_mb: 10.0,
      speed_mb_s: 1.5,
      current_file: 'weights.safetensors',
      stage_message: 'Downloading...',
    };
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockProgress,
    });

    const { fetchModelProgress } = await import('./api');
    const result = await fetchModelProgress();
    expect(result).toEqual(mockProgress);
  });

  it('comparePrompts sends POST request with prompt_a and prompt_b', async () => {
    const mockComparisonResult = {
      prompt_a_label: 'code.md',
      prompt_b_label: 'philosophie.md',
      matrix_a: [[10, 90]],
      matrix_b: [[60, 40]],
      diff_matrix: [[-50, 50]],
      global_jsd: 0.2854,
      layer_jsd: [0.2854],
      top_divergent_experts: [
        {
          layer_id: 0,
          expert_id: 1,
          expert_name: 'Layer 0 - Expert 1',
          activation_a: 90,
          activation_b: 40,
          diff: 50,
          abs_diff: 50,
          dominant_prompt: 'A',
        },
      ],
      num_layers: 1,
      experts_per_layer: 2,
      model_id: 'test-model',
    };

    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockComparisonResult,
    });

    const { comparePrompts } = await import('./api');
    const result = await comparePrompts('code.md', 'philosophie.md');

    expect(result).toEqual(mockComparisonResult);
    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/compare-prompts',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt_a: 'code.md', prompt_b: 'philosophie.md' }),
      })
    );
  });

  it('comparePrompts throws error when backend responds with error', async () => {
    (fetch as any).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: 'Le prompt A ne peut pas être vide' }),
    });

    const { comparePrompts } = await import('./api');
    await expect(comparePrompts('', 'philosophie.md')).rejects.toThrow('Le prompt A ne peut pas être vide');
  });
});

