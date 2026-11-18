# Day08 - Formal LoRA SFT Run Ledger

## 1. Run identity

- Base model: Qwen3-1.7B-Base
- Training data: data/processed/sft.jsonl
- Number of SFT samples: 2000
- Config: configs/sft_full.json
- Output directory: outputs/sft_full
- Final adapter: outputs/sft_full/final_adapter
- Random seed: 42

## 2. LoRA configuration

- r: 16
- alpha: 32
- dropout: 0.05
- bias: none
- target modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

## 3. Training configuration

- per-device batch size: 2
- gradient accumulation steps: 1
- effective batch size: 2
- learning rate: 2e-4
- max steps: 1000
- max length: 512
- completion-only loss: true
- logging steps: 10
- checkpoint save steps: 250

## 4. Actual training result

- global step: 1000
- train loss: 0.4587279015779495
- train runtime: 207.8813 seconds
- train samples/sec: 9.621
- train steps/sec: 4.81
- peak GPU allocated: 8.4903 GB
- peak GPU reserved: 20.1113 GB
- resume from checkpoint: no

## 5. Evaluation

- Base dev-500 strict accuracy: 53.40%
- SFT dev-500 strict accuracy: 78.00%
- Base format success rate: 66.20%
- SFT format success rate: 99.60%

## 6. Adapter identity

- SHA256:
  63420f32629bbf452b30b4e5d27060c924f01b37dea013a073bb2956cea2437f

## 7. Notes

The formal SFT run started from Qwen3-1.7B-Base with a fresh LoRA adapter.
The 20-step smoke run was used only for validation and was not continued into this formal run.
The improvement in strict dev accuracy should not be interpreted entirely as pure reasoning improvement because format failures dropped substantially after SFT.

## 8. Environment

- Platform: Alibaba Cloud PAI-DSW
- GPU: NVIDIA A10
- Python: 3.11.11
- PyTorch: 2.6.0+cu124
- Transformers: 4.51.3
- PEFT: 0.15.2
- TRL: 0.17.0
- Accelerate: 1.6.0

## 9. Compute and cost notes

- Formal SFT training runtime: 207.8813 seconds
- Formal SFT training GPU-hours: approximately 0.0577 GPU-hours
- Cloud instance billing time: not yet recorded
- Storage cost: not yet recorded
- Total cloud cost: not yet recorded

Note: training GPU-hours are not equivalent to total cloud billing time.
The final cloud cost must be taken from the actual PAI-DSW billing record rather than inferred from training runtime.
