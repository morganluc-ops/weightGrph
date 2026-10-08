import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchDomains, fetchDomainAnalysis } from './api';

global.fetch = vi.fn();

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
});
