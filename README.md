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


Results, Plots and logs in respective folders[https://github.com/diptiman-mohanta/State-Space-Models-for-Large-Language-Models/tree/main/diag-ssm-charlm].

Command for running an Experiment
- **Step 1: Configure dataset paths:** Before running any experiment, update the train, validation, and test dataset paths and othe parameters in the configuration files located at [https://github.com/diptiman-mohanta/State-Space-Models-for-Large-Language-Models/tree/main/diag-ssm-charlm/configs].
- **Step 2: Run experiments using predefined configurations:** You can launch experiments at different model scales using the following commands:
```python
python ssm_charlm.py @configs/large.yaml
python ssm_charlm.py @configs/medium.yaml
python ssm_charlm.py @configs/small.yaml
python ssm_charlm.py @configs/tiny.yaml
```
or 
```python
python ssm_charlm.py --train_text "path/to/train/text/file" --val_text "path/to/valid/text/file" --test_text "path/to/train/text/file" --seq_len 128 --batch_size 32 --emb_dim 64 --ssm_hidden 128 --n_layers 2 --dropout 0.1 --epochs 20 --lr 5e-4 --weight_decay 1e-3 --ckpt_dir "experiments/tiny"
```
- **Step 3: Override configuration values from the command line:** Specific configuration parameters (e.g., number of epochs, checkpoint directory, or dataset paths) can be overridden directly via command-line arguments:
```python
python ssm_charlm.py @configs/large.yaml --epochs 150 --ckpt_dir "experiments/large_extended"
```