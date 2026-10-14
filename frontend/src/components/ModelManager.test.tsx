import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ModelManager } from './ModelManager';
import { MoEModelOption, ModelInfo } from '../types';
import * as api from '../api';

vi.mock('../api', () => ({
  fetchModelProgress: vi.fn(),
  uploadLocalModel: vi.fn(),
}));

describe('ModelManager Component', () => {
  const mockModels: MoEModelOption[] = [
    {
      id: 'deepseek-ai/DeepSeek-V2-Lite',
      name: 'DeepSeek V2 Lite',
      architecture: 'DeepSeek-V2',
      num_layers: 16,
      experts_per_layer: 64,
      active_experts_per_tok: 6,
      size_hint: '1.2 Go',
      is_real: true,
    },
    {
      id: 'Qwen/Qwen1.5-MoE-A2.7B',
      name: 'Qwen 1.5 MoE',
      architecture: 'QwenMoE',
      num_layers: 24,
      experts_per_layer: 60,
      active_experts_per_tok: 4,
      size_hint: '2.7 Go',
      is_real: true,
    },
  ];

  const mockCurrentModel: ModelInfo = {
    model_name: 'DeepSeek V2 Lite',
    model_id: 'deepseek-ai/DeepSeek-V2-Lite',
    architecture: 'DeepSeek-V2',
    device: 'cpu',
    num_layers: 16,
    experts_per_layer: 64,
    total_experts: 1024,
    active_experts_per_tok: 6,
    is_real_model: true,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders active model badge and places help tooltip directly beside it', () => {
    render(
      <ModelManager
        models={mockModels}
        currentModel={mockCurrentModel}
        onSelectModel={vi.fn()}
        loadingModel={false}
      />
    );

    const badge = screen.getByTestId('active-model-badge');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent(/DeepSeek-V2-Lite/);

    const helpTooltip = screen.getByTestId('model-help-tooltip');
    expect(helpTooltip).toBeInTheDocument();
    expect(helpTooltip).toHaveTextContent('?');

    // Both badge and tooltip share the same parent container in the top-right
    expect(badge.parentElement).toBe(helpTooltip.parentElement);
  });

  it('renders local folder browse button and hidden webkitdirectory input without single file button', () => {
    render(
      <ModelManager
        models={mockModels}
        currentModel={mockCurrentModel}
        onSelectModel={vi.fn()}
        loadingModel={false}
      />
    );

    const folderButton = screen.getByTestId('browse-folder-button');
    const folderInput = screen.getByTestId('local-folder-input');

    expect(folderButton).toBeInTheDocument();
    expect(folderButton).toHaveTextContent(/Parcourir dossier/i);
    expect(folderInput).toHaveAttribute('type', 'file');

    // Single file button should not exist
    expect(screen.queryByTestId('browse-file-button')).not.toBeInTheDocument();
    expect(screen.queryByTestId('local-file-input')).not.toBeInTheDocument();
  });

  it('triggers folder input click when browse folder button is clicked', () => {
    render(
      <ModelManager
        models={mockModels}
        currentModel={mockCurrentModel}
        onSelectModel={vi.fn()}
        loadingModel={false}
      />
    );

    const folderInput = screen.getByTestId('local-folder-input');
    const clickSpy = vi.spyOn(folderInput, 'click');

    const folderButton = screen.getByTestId('browse-folder-button');
    fireEvent.click(folderButton);

    expect(clickSpy).toHaveBeenCalled();
  });

  it('uploads folder files to server and selects model using returned server path', async () => {
    const handleSelectModel = vi.fn();
    const serverModelPath = '/app/models/my-custom-moe-model';

    (api.uploadLocalModel as any).mockResolvedValueOnce({
      status: 'success',
      model_id: serverModelPath,
      model_name: 'my-custom-moe-model',
      model_path: serverModelPath,
      files_count: 2,
    });

    render(
      <ModelManager
        models={mockModels}
        currentModel={mockCurrentModel}
        onSelectModel={handleSelectModel}
        loadingModel={false}
      />
    );

    const folderInput = screen.getByTestId('local-folder-input');
    const customInput = screen.getByTestId('custom-model-input') as HTMLInputElement;

    const mockConfigFile = new File(['{"model_type": "moe"}'], 'config.json', { type: 'application/json' });
    Object.defineProperty(mockConfigFile, 'webkitRelativePath', {
      value: 'my-custom-moe-model/config.json',
      writable: false,
    });

    const mockWeightFile = new File(['fake-weights'], 'model.safetensors', { type: 'application/octet-stream' });
    Object.defineProperty(mockWeightFile, 'webkitRelativePath', {
      value: 'my-custom-moe-model/model.safetensors',
      writable: false,
    });

    fireEvent.change(folderInput, {
      target: {
        files: [mockConfigFile, mockWeightFile],
      },
    });

    await waitFor(() => {
      expect(api.uploadLocalModel).toHaveBeenCalledTimes(1);
      expect(customInput.value).toBe(serverModelPath);
      expect(handleSelectModel).toHaveBeenCalledWith(serverModelPath);
    });
  });

  it('displays error alert when folder upload fails', async () => {
    (api.uploadLocalModel as any).mockRejectedValueOnce(
      new Error('Extension de fichier non autorisée')
    );

    render(
      <ModelManager
        models={mockModels}
        currentModel={mockCurrentModel}
        onSelectModel={vi.fn()}
        loadingModel={false}
      />
    );

    const folderInput = screen.getByTestId('local-folder-input');
    const mockFile = new File(['bad'], 'script.exe', { type: 'application/octet-stream' });
    Object.defineProperty(mockFile, 'webkitRelativePath', {
      value: 'bad-model/script.exe',
      writable: false,
    });

    fireEvent.change(folderInput, {
      target: {
        files: [mockFile],
      },
    });

    await waitFor(() => {
      const errorAlert = screen.getByTestId('upload-error');
      expect(errorAlert).toBeInTheDocument();
      expect(errorAlert).toHaveTextContent(/Extension de fichier non autorisée/i);
    });
  });

  it('disables controls during model loading', () => {
    render(
      <ModelManager
        models={mockModels}
        currentModel={mockCurrentModel}
        onSelectModel={vi.fn()}
        loadingModel={true}
      />
    );

    expect(screen.getByTestId('browse-folder-button')).toBeDisabled();
    expect(screen.getByTestId('load-model-button')).toBeDisabled();
    expect(screen.getByTestId('load-model-button')).toHaveTextContent(/Téléchargement en cours/i);
  });
});
