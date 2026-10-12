export interface ExpertUsage {
  layer_id: number;
  expert_id: number;
  usage_percentage: number;
  num_params: string;
  expert_name: string;
}

export interface TokenRoutingInfo {
  token: string;
  layer_id: number;
  expert_id: number;
  confidence: number;
}

export interface DomainAnalysisResult {
  domain_id: string;
  domain_name: string;
  experts: ExpertUsage[];
  detected_domains?: { [domainId: string]: number };
  is_real_inference?: boolean;
  model_id?: string;
  token_routing?: TokenRoutingInfo[];
}

export interface DomainsMap {
  [domainId: string]: string;
}

export interface ModelInfo {
  model_name: string;
  model_id?: string;
  architecture: string;
  num_layers: number;
  experts_per_layer: number;
  total_experts: number;
  active_experts_per_tok: number;
  is_real_model?: boolean;
  device?: string;
}

export interface MoEModelOption {
  id: string;
  name: string;
  architecture: string;
  num_layers: number;
  experts_per_layer: number;
  active_experts_per_tok: number;
  size_hint: string;
  is_real: boolean;
  active?: boolean;
}
