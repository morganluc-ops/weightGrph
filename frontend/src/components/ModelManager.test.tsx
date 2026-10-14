import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ModelManager } from './ModelManager';
import { MoEModelOption, ModelInfo } from '../types';

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

  it('renders local folder and file browse buttons and hidden file inputs', () => {
    render(
      <ModelManager
        models={mockModels}
        currentModel={mockCurrentModel}
        onSelectModel={vi.fn()}
        loadingModel={false}
      />
    );

    const folderButton = screen.getByTestId('browse-folder-button');
    const fileButton = screen.getByTestId('browse-file-button');
    const folderInput = screen.getByTestId('local-folder-input');
    const fileInput = screen.getByTestId('local-file-input');

    expect(folderButton).toBeInTheDocument();
    expect(folderButton).toHaveTextContent(/Parcourir dossier/i);
    expect(fileButton).toBeInTheDocument();
    expect(fileButton).toHaveTextContent(/Fichier/i);

    expect(folderInput).toHaveAttribute('type', 'file');
    expect(fileInput).toHaveAttribute('type', 'file');
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

  it('fills custom model input when a local directory is chosen', async () => {
    const handleSelectModel = vi.fn();
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

    const mockFile = new File(['dummy-content'], 'config.json', { type: 'application/json' });
    Object.defineProperty(mockFile, 'webkitRelativePath', {
      value: 'my-custom-moe-model/config.json',
      writable: false,
    });

    fireEvent.change(folderInput, {
      target: {
        files: [mockFile],
      },
    });

    expect(customInput.value).toBe('my-custom-moe-model');

    // Clicking load loads the custom model
    const loadButton = screen.getByTestId('load-model-button');
    fireEvent.click(loadButton);

    expect(handleSelectModel).toHaveBeenCalledWith('my-custom-moe-model');
  });

  it('fills custom model input with full path if available (e.g. Electron / local runtime)', () => {
    const handleSelectModel = vi.fn();
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

    const mockFile = new File(['dummy-content'], 'config.json', { type: 'application/json' });
    Object.defineProperty(mockFile, 'path', {
      value: 'C:\\Models\\local-moe\\config.json',
      writable: false,
    });
    Object.defineProperty(mockFile, 'webkitRelativePath', {
      value: 'local-moe/config.json',
      writable: false,
    });

    fireEvent.change(folderInput, {
      target: {
        files: [mockFile],
      },
    });

    expect(customInput.value).toBe('C:\\Models\\local-moe');
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
    expect(screen.getByTestId('browse-file-button')).toBeDisabled();
    expect(screen.getByTestId('load-model-button')).toBeDisabled();
    expect(screen.getByTestId('load-model-button')).toHaveTextContent(/Téléchargement en cours/i);
  });
});
