export interface ExpertUsage {
  layer_id: number;
  expert_id: number;
  usage_percentage: number;
  num_params: string;
  expert_name: string;
}

export interface DomainAnalysisResult {
  domain_id: string;
  domain_name: string;
  experts: ExpertUsage[];
}

export interface DomainsMap {
  [domainId: string]: string;
}

export interface ModelInfo {
  model_name: string;
  architecture: string;
  num_layers: number;
  experts_per_layer: number;
  total_experts: number;
  active_experts_per_tok: number;
}
