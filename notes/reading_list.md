# Reading list

On 15 Oct 2026 every arXiv entry below was checked through the arXiv API: the ID exists and the title, authors and date match. The **Read** column says how much of each source we have actually read:

- *full*: read in full;
- *code*: source code read;
- *§…*: the listed sections read in full;
- *scanned*: full text searched for specific content (e.g. batch-size and momentum experiments), not read end to end;
- *abstract*: abstract only.

Read statuses were last updated on 19 Oct 2026 (novelty check, see `novelty_check_2026-10-19.md`).

Do not cite a claim from a source we have only read the abstract of until someone reads the relevant section.

## A. Course starter kit (from the official project page)

| Source | Read | Why it matters |
|---|---|---|
| Jordan et al. (2024), *Muon: An optimizer for hidden layers in neural networks*, https://kellerjordan.github.io/posts/muon/ | full | Definition, Newton–Schulz design, FLOP overhead T·m/B, which parameters get Muon |
| Bernstein & Newhouse (2024), *Old Optimizer, New Norm: An Anthology*, arXiv:2409.20325 | abstract | Adam, Shampoo and Muon as steepest descent under different norms |
| Liu et al. (2025), *Muon is Scalable for LLM Training*, arXiv:2502.16982 | abstract | Weight decay, update-RMS matching (0.2·√max(A,B)), about 2× compute efficiency |
| Li et al. (2025), *NorMuon: Making Muon more efficient and scalable*, arXiv:2510.05491 | abstract | Per-neuron normalisation after orthogonalisation |
| Ahn et al. (2025), *Dion: Distributed Orthonormalized Updates*, arXiv:2504.05295 | abstract | Low-rank power iteration with error feedback |

## B. Implementations

| Source | Read | Notes |
|---|---|---|
| github.com/KellerJordan/Muon, `muon.py` (master, last commit 24 May 2026) | code | lr 0.02, weight decay 0, bf16 Newton–Schulz, √max(1, out/in) scaling, flattens convolution weights |
| PyTorch `torch/optim/_muon.py` (v2.9.0, v2.14.1 and main) | code | lr 1e-3, weight decay 0.1 by default; 2-D parameters only; `adjust_lr_fn` is `original`, `match_rms_adamw`, or (not in 2.9) `spectral_unclamped` |
| Optax `optax/contrib/_muon.py` (main; release 0.2.8) | code | By default applies Muon to *all* 2-D parameters; the Adam branch reuses the Muon lr; fp32 Newton–Schulz; bias-corrected momentum |
| github.com/microsoft/dion | README | Reference implementations of Muon, NorMuon and Dion/Dion2/Dion3 |

## C. Benchmarks and scale

| Source | Read | Why it matters |
|---|---|---|
| Essential AI (2025), *Practical Efficiency of Muon for Pretraining*, arXiv:2505.02222 | scanned | Muon keeps its data efficiency at large batch sizes |
| Wen, Hall, Ma & Liang (2025), *Fantastic Pretraining Optimizers and Where to Find Them*, arXiv:2509.02046 | abstract | With fair tuning, speedup shrinks from 1.4× (0.1B parameters) to 1.1× (1.2B) |
| Semenov, Pagliardini & Jaggi (2025), *Benchmarking Optimizers for LLM Pretraining*, arXiv:2509.01440 | abstract | Standardised benchmark |
| Kimi Team (2025), *Kimi K2: Open Agentic Intelligence*, arXiv:2507.20534 | abstract | MuonClip at 1T-parameter scale |
| Khona et al. (2026), *SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales*, arXiv:2607.20548 | abstract | Large-batch stability; checks orthogonalisation quality |
| Mehta et al. (2025), *Muon: Training and Trade-offs with Latent Attention and MoE*, arXiv:2509.24406 | abstract | Small and medium decoders (30M–200M parameters) |

## D. Theory and geometry

| Source | Read | Why it matters |
|---|---|---|
| Bernstein & Newhouse (2024), *Modular Duality in Deep Learning*, arXiv:2410.21265 | abstract | Duality maps; Newton–Schulz for Linear and Conv2D layers |
| Large et al. (2024), *Scalable Optimization in the Modular Norm*, arXiv:2405.14813 | abstract | Why embeddings need a different norm |
| Yang, Simon & Bernstein (2023), *A Spectral Condition for Feature Learning*, arXiv:2310.17813 | abstract | Basis for the √(fan-out/fan-in) scaling |
| Pethick et al. (2025), *Training Deep Learning Models with Norm-Constrained LMOs*, arXiv:2502.07529 | abstract | Linear-minimisation-oracle view (Scion) |
| Kovalev (2025), *Understanding Gradient Orthogonalization … Non-Euclidean Trust-Region*, arXiv:2503.12645 | abstract | Trust-region view, convergence |
| Kim & Oh (2026), *Convergence of Muon with Newton-Schulz*, arXiv:2601.19156 (ICLR 2026) | abstract | Newton–Schulz Muon converges like exact-SVD Muon, up to a constant |
| Shen et al. (2025), *On the Convergence Analysis of Muon*, arXiv:2505.23737 | abstract | When Muon beats gradient descent |
| Chen, Li & Liu (2025), *Muon Optimizes Under Spectral Norm Constraints*, arXiv:2506.15054 | abstract | Implicit spectral-norm constraint from weight decay |
| Sfyraki & Wang (2025), *Lions and Muons: … Stochastic Frank-Wolfe under Heavy-Tailed Noise*, arXiv:2506.04192 (ICML 2026) | abstract | Frank–Wolfe view; heavy-tailed noise |
| Fan, Schmidt & Thrampoulidis (2025), *Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data*, arXiv:2502.04664 (NeurIPS 2025) | abstract | Max-margin in the spectral norm |
| Su (2025), *Isotropic Curvature Model … Is Gradient Orthogonalization Optimal?*, arXiv:2511.00674 | abstract | Orthogonalisation points the right way but may not be optimal |
| Davis & Drusvyatskiy (2025), *When do spectral gradient updates help in deep learning?*, arXiv:2512.04299 | abstract | Layer-wise test: nuclear-to-Frobenius ratio of the gradient vs stable rank of the activations |
| Sato, Naganuma & Iiduka (2025), *Convergence Bound and Critical Batch Size of Muon Optimizer*, arXiv:2507.01598 | §5 | Critical batch size; ResNet/VGG on CIFAR experiments |
| Gonon, Muşat & Boumal (2026), *Insights on Muon from Simple Quadratics*, arXiv:2602.11948 | abstract | Errors in the polar step can *help* finite-time behaviour |
| Peyré (2026), *Muon Dynamics as a Spectral Wasserstein Flow*, arXiv:2604.04891 | abstract | Mean-field view; Schatten-norm interpolation |

## E. Newton–Schulz accuracy and cost

| Source | Read | Why it matters |
|---|---|---|
| Amsel, Persson, Musco & Gower (2025), *The Polar Express*, arXiv:2505.16932 | abstract | Coefficients that are optimal at each step; works in bf16 |
| Grishina, Smirnov & Rakhuba (2025), *Accelerating Newton-Schulz … Chebyshev-type Polynomials* (CANS), arXiv:2506.10935 | abstract | Chebyshev-optimised coefficients |
| Boissin et al. (2025), *Turbo-Muon*, arXiv:2512.04632 | abstract | Preconditioning that saves one Newton–Schulz step |
| Shulgin, AlRashed, Orabona & Richtárik (2025), *Beyond the Ideal: Analyzing the Inexact Muon Update*, arXiv:2510.19933 | abstract | Lower precision calls for a smaller lr and larger momentum |
| Huang (2026), *How Much Orthogonalization Does Muon Need?*, arXiv:2606.00371 | abstract | Training quality is not monotone in orthogonalisation accuracy |
| Magakyan, Parrilo & Ozdaglar (2026), *Spectral Scaling Laws of Muon*, arXiv:2606.04058 | abstract | Small singular values of the momentum fail to be orthonormalised |
| Li & Tsuchiya (2026), *Muon with Finite Newton-Schulz: The Smoothing Benefit*, arXiv:2608.26288 | abstract | Theory: a finite number of Newton–Schulz steps can help |
| Choudhury et al. (2026), *Muon with Nesterov Momentum: Heavy-Tailed Noise and (Randomized) Inexact Polar Decomposition*, arXiv:2605.06884 | abstract | Theory with inexact polar decomposition |

## F. Signal vs noise in weak directions, and when Muon helps (closest to our provisional direction)

| Source | Read | Why it matters |
|---|---|---|
| Goyal & Ramkumar (2026), *Spectrally Targeted Muon*, arXiv:2610.10965 | full | "Small singular values of the momentum are not noise" (CIFAR-10 and NanoGPT speedruns) |
| Xie (2026), *Muon Under Gradient Noise and the Limits of Orthogonalization Near Optima*, arXiv:2609.32861 | §1–3, §7–8 (no proofs) | Theory: near an optimum, noise raises Muon's loss floor; switch to matched momentum-SGD |
| Jacobs & Burkholz (2026), *From Parallel to Sequential: Small-Batch Noise Restores Simplicity Bias of Muon*, arXiv:2610.16868 | §1–2, §6–7 | Small batches change how Muon learns features |
| Li, Zhang, Liu & Bao (2026), *Denoise First, Orthogonalize Later*, arXiv:2606.03899 | §1, App. F.5 | Momentum acts as a spectral filter before orthogonalisation |
| Dong & Sawin (2026), *Muon^p: Muon with Fractional Spectral Powers*, arXiv:2606.13867 | scanned | Partial flattening U S^p Vᵀ |
| Pang et al. (2026), *HTMuon: Improving Muon via Heavy-Tailed Spectral Correction*, arXiv:2603.10067 | scanned | Muon over-weights noise-dominated directions |
| Kim et al. (2026), *AMUSE: Anytime Muon with Stable Gradient Evaluation*, arXiv:2605.22432 | abstract | Muon amplifies noise in dominant directions |
| Beneventano, Abdelmoneum & Poggio (2026), *The Spectral Dynamics and Noise Geometry of Muon*, arXiv:2606.08388 | §1, §5.5, §6–8 | Depends on regime: beats AdamW on NanoGPT, loses on a small ViT |
| Dragutinović, Zhang & Ranganath (2026), *To Use or not to Use Muon*, arXiv:2603.00742 | abstract | Loses the simplicity bias of gradient descent |
| Nguyen et al. (2026), *Spectral Flattening Is All Muon Needs*, arXiv:2605.13079 | abstract | Stable step size scales with the average singular value |
| Vasudeva et al. (2025), *How Muon's Spectral Design Benefits Generalization: A Study on Imbalanced Data*, arXiv:2510.22980 | abstract | Learns all principal components at equal rates |
| Wang et al. (2025), *Muon Outperforms Adam in Tail-End Associative Memory Learning*, arXiv:2509.26030 | abstract | Heavy-tailed data |
| Southworth et al. (2026), *Muon in Vision Transformers*, arXiv:2605.24770 | abstract | Gradient spectra span 2–4 orders of magnitude |
| Ruan et al. (2026), *Muon Learns More Robust and Transferable Features than Adam*, arXiv:2606.09658 | abstract | Feature quality |
| Liu, Paquette & Sous (2026), *Spectral Lens*, arXiv:2605.05683 | abstract | Spectra as training diagnostics; batch size shapes representations |
| Tveit, Remseth & Skogvold (2025), *Muon Optimizer Accelerates Grokking*, arXiv:2504.16041 | abstract | Small-task example with statistical testing |

## G. Variants

| Source | Read |
|---|---|
| Si, Zhang & Shen (2025), *AdaMuon*, arXiv:2507.11005 | abstract |
| Lau, Long & Su (2025), *PolarGrad*, arXiv:2505.21799 | abstract |
| Riabinin et al. (2025), *Gluon*, arXiv:2505.13416 | abstract |
| Gruntkowska et al. (2025), *Error Feedback for Muon and Friends*, arXiv:2510.00643 | abstract |
| Newhouse et al. (2025), *Training Transformers with Enforced Lipschitz Constants*, arXiv:2507.13338 | abstract |

## H. Background

| Source | Read |
|---|---|
| Gupta, Koren & Singer (2018), *Shampoo*, arXiv:1802.09568 | abstract |
| Vyas et al. (2024), *SOAP*, arXiv:2409.11321 | abstract |
| Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization*, arXiv:1711.05101 (ICLR 2019) | abstract |
| Tuddenham, Prügel-Bennett & Hare (2022), *Orthogonalising gradients to speed up neural network optimisation*, arXiv:2202.07052 | abstract |
| Bernstein (2025), *Deriving Muon*, https://jeremybernste.in/writing/deriving-muon | not yet read (cited by PyTorch's docs and Jordan's README) |
| Higham (2008), *Functions of Matrices* (SIAM); Björck & Bowie (1971); Kovarik (1970) | not yet read (classical Newton–Schulz references) |

## I. Added on 19 Oct 2026 (novelty check)

| Source | Read | Why it matters |
|---|---|---|
| Dang, Wen & Malladi (2026), *The Best Optimizer Depends on Batch Size*, arXiv:2610.08975 | abstract, §1, scaling-rule table | Optimizer rankings (incl. Muon) cross with batch size after per-batch tuning; no Muon scaling rule works everywhere |
| Wu, Shah, Silwal & Zhang (2026), *DynMuon: A Dynamic Spectral Shaping View of Muon*, arXiv:2605.17109 | §2–4 | Late training prefers mildly negative p (−0.25); tested at one large batch only |
| Shumaylov et al. (2026), *Muon is Not That Special: Random or Inverted Spectra Work Just as Well*, arXiv:2605.11181 | abstract | Exact spectral shape matters little; step-size optimality does |
| Massena, Friedrich & Serrurier (2026), *From SGD to Muon: Adaptive Optimization via Schatten-p Norms*, arXiv:2605.19781 (NeurIPS 2026) | abstract | Picks a per-layer geometry between SGD and Muon from gradient and activation statistics |
| Qi, Chen, Ye, He & Xiao (2026), *Delving into Muon and Beyond*, arXiv:2602.04669 | abstract | U S^p Vᵀ family (p = 0, 1/4, 1/2, 1); Muon not consistently better than Adam |
| Fan, Liu, Hong, Kompella & Liu (2026), *Rethinking Muon Beyond Pretraining*, arXiv:2605.19282 | abstract | Muon amplifies noisy tail directions; collapses in low signal-to-noise RL post-training |
| Marek, Lotfi, Somasundaram, Wilson & Goldblum (2025), *Small Batch Size Training for Language Models*, arXiv:2507.07101 (NeurIPS 2025) | abstract | Small batches train stably; even plain SGD works |
| Yi (2026), *MuCon: Clipped Muon Updates*, arXiv:2605.26459 | abstract | Singular-value clipping instead of full flattening |
| Shi et al. (2026), *Pion: A Spectrum-Preserving Optimizer*, arXiv:2605.12492 | abstract | Updates that rotate weights while keeping their singular values |
| Yukhimchuk, Kolar, Takáč & Choudhury (2026), *Gradient Clipping Beyond Vector Norms*, arXiv:2605.11838 | abstract | Spectral clipping of the largest singular values |
| Iiduka (2026), *Muon Converges under Heavy-Tailed Noise*, arXiv:2603.15059 | abstract | Convergence theory under heavy-tailed noise |
| Davis & Drusvyatskiy (2025), arXiv:2512.04299 (listed in D) | downloaded, not yet read | Layer-wise test; candidate diagnostic |
