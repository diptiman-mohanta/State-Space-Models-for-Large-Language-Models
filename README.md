# State-Space-Models-for-Large-Language-Models
## Diagonal SSM Character-Level Language Model on Indic Text

Pure PyTorch implementation of a **diagonal State Space Model (SSM)** for character-level language modeling on Indic text. Inspired by S4 with residual stacking and diagonal recurrence.

This repo contains multiple ablation experiments varying model size, sequence length, batch size, and training duration.

### Architecture

- Embedding → Input projection
- N stacked residual blocks: DiagSSMLayer (diagonal SSM) → LayerNorm → GELU → Dropout → Residual add
- Final linear head to vocab (~74 Devanagari tokens)

Simplified non-selective SSM (fixed diagonal A, input-independent B/C).

### Experiments

| Experiment       | seq_len | batch_size | emb_dim | ssm_hidden | n_layers | epochs | Val PPL (Best) | Test PPL (Ensemble) | Notes                  |
|------------------|---------|------------|---------|------------|----------|--------|----------------|---------------------|------------------------|
| large      | 1024   | 4         | 256    | 768       | 8       | 100   | ~4.09         | 4.08               | Original large model  |
| medium  | 512    | 8         | 128    | 512       | 6       | 50    | 4.39    | -   4.45               | Standard config       |
| small      | 256    | 16        | 64     | 256       | 4       | 100    | 4.66     | 4.68                | Quick training run    |
| tiny      | 128    | 32        | 64     | 128       | 2       | 100    | 6.08     | 6.08                  | code debugging run    |
Results, Plots and logs in respective folders.