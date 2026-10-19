# Novelty check: 19 Oct 2026

**What we checked.** Whether the provisional direction is still open given papers from the past few weeks. That direction asked whether the weak singular directions Muon amplifies are signal or noise, tested by varying batch size × momentum.

**How.** We read the full text (arXiv HTML or PDF) of the closest papers. For each one we recorded the setting, what it varies, its main finding and what it leaves open. The "We read" column lists the sections actually read. No proofs were checked.

## Summary

1. **"Does Muon's advantage change with batch size?" is now well covered.**
   - Dang, Wen & Malladi (6 Oct 2026) tuned five optimizers independently at every batch size: language-model pretraining at 128K–2M tokens per batch. Rankings cross between batch sizes, and no Muon learning-rate scaling rule worked across settings.
   - Sato et al. found Muon ahead of AdamW at every tested batch size on ResNet-18/CIFAR-10, with the gap narrowing at the largest batches on a 320M language model. They tuned the learning rate only at batch 512, then scaled it by √(batch size).
   - On its own, a small-MLP version of this question would add little.
2. **"Are the small singular directions noise?" has an answer, but only for large batches.** Spectrally Targeted Muon (7 Oct 2026) says they carry signal, based on the CIFAR-10 speedrun (batch 500) and the NanoGPT speedrun. It never varies batch size or noise.
3. **Two 2026 papers make opposite predictions about late training.** In the family of updates U S^p V^T applied to the momentum buffer, p = 0 is Muon and p = 1 is plain momentum SGD.
   - **Xie (26 Sep 2026)** gives theory for when gradients are noise-dominated. There, Muon's expected update is a scaled gradient step plus extra fluctuation, so switching to momentum SGD with a matched learning rate (p = 1) lowers the stationary loss.
     - Predicted ratio of excess loss, Muon over SGD: 1.04–1.13 at momentum 0.9–0.95, and about 1.34–1.47 without momentum.
     - Tested only on quadratics and on frozen transformer gradients. In the paper's words: "No experiment trains a network with a switched optimizer."
   - **DynMuon (Wu et al., May 2026)** finds a mildly *negative* p (−0.25), which flattens even more than Muon, helps late in training. The reason given: the remaining signal sits in flat, low-curvature directions.
     - Tested only at one large batch (512 sequences × 1024 tokens).
     - Its own noise model says negative p helps only while that remaining signal is larger than the noise.
   - **Both theories therefore predict that the best late-training p should rise towards 1 as gradient noise rises. Nobody has tested this by varying the noise.**
4. **Caution: step size can masquerade as spectral shape.** Shumaylov et al. (May 2026) report that the exact spectral shape matters little once the step size is right: random or inverted spectra "work just as well". Any comparison across exponents must tune the step size separately for each p, otherwise shape and step size are confounded.

## Paper by paper

| Paper | Setting | What it varies | Main finding (for us) | What it leaves open | We read |
|---|---|---|---|---|---|
| Goyal & Ramkumar (2026), *Spectrally Targeted Muon*, arXiv:2610.10965 | CIFAR-10 CNN speedrun (batch 500, 20 trials); NanoGPT 124M speedrun (2–3 seeds) | Threshold τ selecting which singular values are orthogonalised; learning rates held fixed | Orthogonalising everything except the top few directions ≈ Muon; small singular values are "not noise"; warmup hurts Muon | Batch size and noise not varied; τ relative to each matrix's spectrum; τ changing during training | Full |
| Xie (2026), *Muon Under Gradient Noise and the Limits of Orthogonalization Near Optima*, arXiv:2609.32861 | Theory; Monte Carlo; quadratics; frozen gradients of a 6-layer character-level transformer (Tiny Shakespeare, effective batch 64) | Momentum; NS5 vs exact polar; bf16 vs fp32 | Near the optimum Muon's stationary loss exceeds momentum SGD with a matched learning rate; momentum shrinks the gap but never removes it | Switch never tested in training; detecting the noise-dominated regime; threshold; matched learning rate | §1–3, §7–8 (proofs not read) |
| Wu et al. (2026), *DynMuon*, arXiv:2605.17109 | GPT-style 127M–1.1B and a Qwen-style model; FineWeb; global batch 512 × 1024 tokens | Exponent p and its schedule | Scheduling p from 1 to −0.25 beats Muon: validation loss lower by 0.01–0.02, 10–27% fewer steps to target | Noise level not varied | §2–4 |
| Jacobs & Burkholz (2026), *From Parallel to Sequential*, arXiv:2610.16868 | Deep linear theory; matrix factorisation; DeiT-S/ImageNet; CIFAR-10/100; 162M Llama | Batch size and learning rate, Muon only | Small-batch noise makes Muon's expected step proportional to the gradient, which brings back sequential feature learning and lower effective rank | No comparison with other optimizers or other exponents | §1–2, §6–7 |
| Dang, Wen & Malladi (2026), *The Best Optimizer Depends on Batch Size*, arXiv:2610.08975 | Modded-NanoGPT benchmark (1.7B tokens, batch 128K–2M tokens); ResNet on CIFAR-5M for scaling rules; noisy quadratic model | Batch size, with every optimizer retuned at every batch size | No Muon scaling rule works everywhere; optimizer rankings cross between batch sizes | Small batches, small networks, the spectral exponent | Abstract, §1, scaling-rule table |
| Sato, Naganuma & Iiduka (2025), *Critical Batch Size of Muon*, arXiv:2507.01598 | ResNet-18/CIFAR-10; VGG-16/CIFAR-100; Llama 320M/C4; Fashion-MNIST MLP; teacher-student task | Batch size (32–8192), β, Nesterov, weight decay | Muon needs fewer steps and less compute than AdamW at every batch size on the vision tasks; on C4 the gap narrows at the largest batches; critical batch size falls as β → 1 | Learning rate tuned once at batch 512, then √-scaled | §5 |
| Li et al. (2026), *Denoise First, Orthogonalize Later*, arXiv:2606.03899 | Theory; NanoGPT and LLaMA-350M end to end; frozen probes | Order of momentum and orthogonalisation; batch size (probes only) | Applying momentum before orthogonalisation keeps the update aligned with the signal; its advantage grows as the batch shrinks (probes only) | Batch-size effect in full training | §1, App. F.5 |
| Beneventano et al. (2026), *Spectral Dynamics and Noise Geometry of Muon*, arXiv:2606.08388 | Theory; matrix sensing; NanoGPT (single seed); small ViT | — | The polar map's sensitivity to noise grows when the matrix has small singular values; Muon beats AdamW on NanoGPT and loses on the small ViT | Single seed; the paper says it was largely written by AI, so treat with caution | §1, §5.5, §6–8 |
| Muon^p (2606.13867); HTMuon (2603.10067); Essential AI (2505.02222) | LLM fine-tuning and pretraining | p or tail exponent at a fixed batch; Essential AI also varies batch from 128K to 16M tokens | Partial flattening helps fine-tuning; a heavy-tailed correction helps; Muon is better at very large batches | Whether the best p depends on noise | Scanned for batch-size and momentum content |

Also relevant, abstract only (all verified on arXiv):

- Shumaylov et al., *Muon is Not That Special*, arXiv:2605.11181
- Massena et al., *From SGD to Muon: Adaptive Optimization via Schatten-p Norms*, arXiv:2605.19781 (NeurIPS 2026)
- Qi et al., *Delving into Muon and Beyond*, arXiv:2602.04669
- Fan et al., *Rethinking Muon Beyond Pretraining*, arXiv:2605.19282: Muon amplifies noisy tail directions and collapses in low signal-to-noise RL post-training
- Marek et al., *Small Batch Size Training for Language Models*, arXiv:2507.07101 (NeurIPS 2025)

## Revised research question (provisional, needs group sign-off)

**Late in training, should Muon flatten more or less, and does the answer depend on gradient noise?**

### Hypotheses

- **H1 (main, falsifiable).** Let p\* be the best late-training exponent, with the step size tuned separately for each p. p\* rises with gradient noise:
  - at large batch and high momentum, p\* ≤ 0;
  - at small batch or low momentum, p\* approaches 1 (momentum SGD).
  If p\* does not move with batch size or momentum, H1 is falsified.
- **H2 (Xie's ordering).** The gain from switching late from Muon to momentum SGD with a matched learning rate is larger at lower momentum.
- **H3 (mechanism).** The per-layer noise indicator δ_β = ‖mean gradient‖_F / (σ_β·√(mn)) (Xie, eq. 32) predicts when the switch helps.

### Design sketch

- **Shared phase.** Train with Muon for the first part of training and save a checkpoint. The switch point is still to be decided.
- **Branches.** From each checkpoint, continue with p ∈ {−0.25, 0, 0.5, 1}.
  - Use exact SVD: about 3.8 ms per 256×256 matrix on the M3 CPU (`code/checks/spectral_shaping_cost.py`).
  - Tune the step size for each p over the same small grid.
  - Use the same data order in every branch, so comparisons are paired.
- **Factors.** Batch size {32, 128, 512, 2048} × momentum {0.9, 0.95}; add 0.5 if time allows.
- **Models.**
  - First: an MLP on Fashion-MNIST or CIFAR-10.
  - Replication: a tiny character-level transformer on Tiny Shakespeare. This is the model family Xie used for its frozen-gradient test.
- **Reference runs.** Muon, AdamW and SGD with momentum for the whole run, each tuned per batch size.
- **Readouts.**
  - Final validation loss and accuracy.
  - Paired difference from p = 0, with confidence intervals over seeds.
  - δ_β for each layer over the course of training.
- **Check.** Include ordinary Newton–Schulz Muon at p = 0, since exact-SVD Muon is not the same optimiser.

### What each outcome would mean

| Outcome | Interpretation |
|---|---|
| p\* rises with noise | Reconciles DynMuon and Xie. Practical rule: flatten less when noise dominates. |
| p\* ≈ 0 everywhere | Muon's full flattening is already the right compromise in these networks. |
| p\* < 0 everywhere | DynMuon's late-training rule extends to small batches; Xie's noise floor is not visible at this scale. |
| Differences within seed noise | Report effect sizes and equivalence bounds. That is still an informative negative result. |

### Novelty statement

As of 19 Oct 2026, none of the papers above sweeps Muon's late-training spectral exponent while varying the gradient-noise level in full training. Xie's late switch has not been tested in full training either. This claim covers only the papers we found and read. A final search is due before the report is written.

### Risks

- **Small effects.** DynMuon gains 0.01–0.02 nats at 127M–1.1B parameters. Xie predicts a few percent of the *excess* loss. Mitigations: paired branches and many seeds. The result may still be inconclusive.
- **Step size confounded with shape.** Handled by tuning the step size for each p.
- **Scope versus deadline.** The deadline is still unknown. Plan an MLP-only core with the transformer as optional.
