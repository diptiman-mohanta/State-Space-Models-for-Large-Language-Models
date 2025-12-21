#Date: 1st Dec 2025
---

# Paper 1: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
# Authors: Albert Gu, Tri Dao
## Introduction
-  Foundational models (FMs) or LMs pretrained on massive data then adapted for downstreaming tasks have emerged as an effective paradigm in the mordern ML.
-  the bacnone of these FMs are often seq models operation og arbiratary sequence of inputs from a wide variety of domains.
-  This concep is agnostic to a particular choice of model architecture, mordern FMs are predominantly based on a single type of sequence model that is transformer and its core attention layer. the efficacy of self attention is attributed to its ability to route information densely within the context window allowing it to model complex data.
-  This brings an fundamental in ability to model anything outside of a finite window anda quadratic scaling woth respect to the window length.
-  Now th SSMs have emerged as a promising class of architecutre for seq to seq modeling. this can be interpreted as combination of RNN and CNN with the inspiration from classical state space models.  
- This class of models can be computed very efficiently as either a recurrence or convolution, with linear or non linear scaling in seq length. and also they have principled mechanisms for modeling long range dependencies in certain data modalities. They work fine in continious signal dara sucg as audio and vision but less effective at modeliing discrete and information-dense data suc as text.
- Proposed a New Class of Selective state space models:
    - Selective Mechanism: Identifed the key limitation of prior models the ability to efficiently select data in an input dependent manner. Designed a simple selection mechanism by parameterizing the SSM parameters nad function based on the input. This allows the model to filter out irrelevant information and remember relevent information indefinitely.
    - Hardware-aware Algorithm: They design a recurrent-mode algorithm that's hardware-aware, using a scan (not convolution) without materializing the full expanded state, avoiding GPU memory hierarchy issues (e.g., IO between SRAM and HBM). This is theoretically linear in sequence length (vs. pseudo-linear for convolutions) and 3× faster on A100 GPUs.
    - Architecture: They simplify prior SSM designs by combining them with Transformer MLP blocks into a single, homogeneous block called Mamba, incorporating selective SSMs without attention.
 - Mamba is a "fully recurrent model" with: (i) High quality on dense modalities (e.g., language, genomics); (ii) Fast training/inference (linear scaling, constant-time autoregressive steps, no KV cache); (iii) Long context handling (performance improves up to 1M sequences).
 - They have tested the Mamba in different modalities:
     - Synthetics: one imp synthetic task such as copying and inducion head have been proposed as being key to LLMs mamba solve them easily but can extrapolate solutions indefinitely long.
     - Audio and Genomics: it outperforms the SSM models such as SaShiMi, Hyena and Transformers on modeling audio waveforms.
     - Language modelling: Mamba is the first linear time sequence mdoel that acives transformer both in pretraining perplexity and downstream evaluation. With scaling into 1B parameters it outperform LLaMa .
  
## State Space Models
- Structured State Space Sequence Models (S4) are a recent class of sequence models for deep learning that are boardly related to RNNs and CNNs and classical state space models.
- They are inspired by a particular continious system that maps a 1-D function or seq through an implicit latent state.
- S4 models are defined with four parameters (Δ,A,B,C) which defines a seq to seq transformation in two stages.
---

