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

export interface LayerBalanceMetrics {
  layer_id: number;
  entropy: number;
  gini_coefficient: number;
  dead_experts_count: number;
  total_experts: number;
  max_expert_load_share: number;
}

export interface BalanceMetrics {
  entropy: number;
  gini_coefficient: number;
  dead_experts_count: number;
  total_experts: number;
  max_expert_load_share: number;
  layers_metrics: LayerBalanceMetrics[];
}

export interface DomainAnalysisResult {
  domain_id: string;
  domain_name: string;
  experts: ExpertUsage[];
  detected_domains?: { [domainId: string]: number };
  is_real_inference?: boolean;
  model_id?: string;
  token_routing?: TokenRoutingInfo[];
  balance_metrics?: BalanceMetrics;
  layer_expert_matrix?: number[][];
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

export interface DownloadProgressInfo {
  status: 'idle' | 'downloading' | 'loading_tensors' | 'ready' | 'error';
  model_id: string;
  downloaded_bytes: number;
  total_bytes: number;
  downloaded_mb: number;
  total_mb: number;
  progress_percent: number;
  speed_mb_s: number;
  current_file: string;
  stage_message: string;
}

export interface DivergentExpert {
  layer_id: number;
  expert_id: number;
  expert_name: string;
  activation_a: number;
  activation_b: number;
  diff: number;
  abs_diff: number;
  dominant_prompt: 'A' | 'B' | 'neutral';
}

export interface PromptComparisonResult {
  prompt_a_label: string;
  prompt_b_label: string;
  matrix_a: number[][];
  matrix_b: number[][];
  diff_matrix: number[][];
  global_jsd: number;
  layer_jsd: number[];
  top_divergent_experts: DivergentExpert[];
  num_layers: number;
  experts_per_layer: number;
  model_id: string;
}

export interface PromptComparisonRequest {
  prompt_a: string;
  prompt_b: string;
}

