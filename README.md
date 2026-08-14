# Awesome Muon Optimizer [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated guide to **Muon** and closely related matrix-aware, orthogonalized, and
spectral optimization research: what the method is, where its ideas came from, how
the literature has developed, what the evidence does and does not support, and what
remains unresolved.

- **Last updated:** 2026-08-14
- **Research coverage through:** 2026-08-14
- **Scope:** this is a *curated* collection, not an exhaustive index. Papers are
  included when Muon is the subject, a directly modified component, a central
  experimental variable, or a necessary part of the intellectual lineage. See
  [CONTRIBUTING.md](CONTRIBUTING.md) for the inclusion standard.

Supportive, mixed, and critical results are listed together and labelled. Venue
labels appear only where an official venue page or the arXiv `Comments` field
confirms them; everything else is marked `Preprint`.

For the long-form historical and scientific synthesis, see
**[docs/research-landscape.md](docs/research-landscape.md)**.

## Contents

- [What is Muon?](#what-is-muon)
- [Research Landscape at a Glance](#research-landscape-at-a-glance)
- [How the Research Evolved](#how-the-research-evolved)
- [Suggested Reading Paths](#suggested-reading-paths)
- [Recent Additions](#recent-additions)
- [Papers by Topic](#papers-by-topic)
  - [Origins and Foundational Perspectives](#origins-and-foundational-perspectives)
  - [Theory, Convergence, and Implicit Bias](#theory-convergence-and-implicit-bias)
  - [Scaling, Parameterization, and Critical Batch Size](#scaling-parameterization-and-critical-batch-size)
  - [Spectral Shaping, Adaptivity, and Muon Variants](#spectral-shaping-adaptivity-and-muon-variants)
  - [Orthogonalization and Numerical Algorithms](#orthogonalization-and-numerical-algorithms)
  - [Distributed and Communication-Efficient Training](#distributed-and-communication-efficient-training)
  - [Quantization and Memory Efficiency](#quantization-and-memory-efficiency)
  - [Fine-Tuning, Post-Training, and Optimizer Transfer](#fine-tuning-post-training-and-optimizer-transfer)
  - [Empirical Benchmarks and Applications](#empirical-benchmarks-and-applications)
  - [Generalization, Robustness, and Regularization](#generalization-robustness-and-regularization)
  - [Limitations, Counterexamples, and Negative Results](#limitations-counterexamples-and-negative-results)
  - [Adjacent Matrix and Spectral Optimizers](#adjacent-matrix-and-spectral-optimizers)
- [Implementations and Ecosystem](#implementations-and-ecosystem)
- [Blog Posts and Explanatory Resources](#blog-posts-and-explanatory-resources)
- [Contributing](#contributing)
- [Acknowledgements and License](#acknowledgements-and-license)

## What is Muon?

Muon is a first-order optimizer for the *matrix-valued* parameters of a neural
network. Instead of treating a weight matrix as a flat vector of independent
coordinates, it forms a momentum buffer from the matrix-shaped gradient and then
transforms that buffer toward its **polar factor** — the semi-orthogonal matrix
obtained by replacing every singular value with 1 while keeping the singular
vectors. The resulting direction has a flat spectrum, so no single direction in the
update dominates by magnitude alone.

Computing the polar factor exactly would require an SVD at every step. Practical
implementations instead run a small fixed number of **Newton–Schulz** iterations —
matrix-multiplication-only polynomial recurrences that drive the singular values
toward 1 — typically five steps in `bfloat16`. The approximation is deliberately
loose: recent work finds that training quality does not track polar-decomposition
accuracy closely (see [How Much Orthogonalization Does Muon Need?](https://arxiv.org/abs/2606.00371)).

Muon is normally applied only to **eligible parameters**: two-dimensional hidden
weights such as attention and MLP projections. Biases, normalization gains,
embeddings, and output heads are usually handled by a **fallback optimizer**,
almost always AdamW. Implementations differ in ways that matter for reproducing
results — the update scale, whether weight decay is applied, the momentum
convention (Nesterov or not), the Newton–Schulz coefficients and iteration count,
and whether fused QKV matrices are split before orthogonalization. Two update-scale
conventions are common enough to have names in framework code: the original
`sqrt(max(1, A/B))` and Moonshot's RMS-matching `0.2·sqrt(max(A, B))`.

A **schematic** sketch of one step — not the exact form used by any specific
implementation:

```text
  M_t  =  μ · M_{t-1}  +  G_t                      momentum on the matrix gradient
  O_t  ≈  polar(M_t)  =  U Vᵀ      where  M_t = U Σ Vᵀ
  W_t  =  W_{t-1}  −  η · s(W) · O_t
```

`G_t` is the gradient of the loss with respect to an eligible weight matrix `W` at
step `t`; `M_t` is the momentum buffer; `μ` is the momentum coefficient; `U Σ Vᵀ`
is the singular value decomposition of `M_t`; `polar(·)` denotes the polar factor
`U Vᵀ`; `O_t` is the approximate polar factor actually computed by Newton–Schulz;
`η` is the learning rate; and `s(W)` is an implementation-specific scale that
usually depends on the matrix shape. Weight decay, when used, is applied separately
and its placement differs between implementations.

Two terminology notes carried throughout this list. For a rectangular `m × n`
matrix the polar factor is **semi-orthogonal**, not orthogonal — it has orthonormal
rows or columns, whichever is shorter, and cannot have both. And the operation is
frequently called the **matrix sign** function, by analogy with the scalar sign,
which is why "orthogonalization", "polar factor", and "matrix sign" all appear in
the literature for closely related objects.

The canonical primary source is Keller Jordan's original post,
[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/);
there is no standalone paper for the method itself. The reference implementation is
[KellerJordan/Muon](https://github.com/KellerJordan/Muon).

## Research Landscape at a Glance

The "emerging picture" column reflects where the weight of currently available
evidence sits. It is not a claim of consensus.

| Research thread | Central question | Emerging picture | Main open issue |
|---|---|---|---|
| Geometry and implicit bias | Is Muon best understood as steepest descent under the spectral norm? | The norm/LMO framing reproduces the update cleanly and predicts several behaviours; implicit-bias results exist for separable linear settings | At least one ablation matches Muon with random or inverted spectra, so geometry may not be the operative mechanism |
| Convergence and optimization theory | Under what assumptions does Muon converge, and how fast? | Rates exist for nonconvex smooth and heavy-tailed-noise settings, several sharper than comparable SGD rates | Muon provably fails to converge on convex Lipschitz functions and on a class of stochastic problems; the function class that explains its practice is not settled |
| Scaling and parameterization | Does the advantage over AdamW survive to larger models? | Reported speedups are real but shrink with scale and depend heavily on tuning parity and on the unit of measurement | No agreed protocol for a fair AdamW comparison; almost all controlled evidence stops well below frontier scale |
| Orthogonalization accuracy and numerics | How exact does the polar factor need to be? | Cheaper and deliberately less accurate iterations often match or beat the standard quintic schedule | Whether approximation error is benign at frontier scale, where momentum spectra are measured to decay faster |
| Adaptive and partial spectral shaping | Should all singular values be flattened to 1? | Several independent lines report that partial, weighted, or curvature-aware shaping is preferable in specific regimes | No predictive rule for how much shaping to apply, to which matrix, at which stage |
| Batch size and gradient noise | Does Muon change the critical batch size? | Muon appears to retain data efficiency at larger batch sizes than AdamW in the evaluated settings | Batch-size results come from a small number of labs and model families |
| Distributed execution and sharding | Can whole-matrix operations coexist with parameter sharding? | Multiple working designs now exist — low-rank, block-periodic, sampled, and ownership-based | Systems papers frequently report throughput without matched quality evidence, and vice versa |
| Quantization and optimizer-state memory | How compressible is Muon's state? | 8-bit is near-lossless in the evaluated settings and 4-bit is reported workable with directional protection | Interaction between quantization noise and orthogonalization is only beginning to be analyzed |
| Fine-tuning and optimizer transfer | Do pretraining conclusions transfer to post-training? | Switching an Adam-pretrained model to Muon is frequently harmful unless the update is constrained | The mechanism behind mismatch severity is unidentified, and RL/post-training evidence is thin |
| Evaluation methodology | How much of the reported advantage is a tuning artefact? | Under equal-density tuning the gap narrows substantially, and controlled problems sometimes erase it | Reporting units — steps, tokens, FLOPs, wall-clock — are not comparable across papers |

## How the Research Evolved

A condensed narrative. The full version, with the evidence behind each claim, is in
[docs/research-landscape.md](docs/research-landscape.md).

**1. Origins (2015–2024).** Spectral-norm steepest descent for deep learning
predates Muon by nearly a decade, and orthogonalizing gradients before an SGD step
had been tried directly. Separately, a line of work on metrizing networks —
modular norms, duality maps, and the observation that many familiar optimizers are
steepest descent under *some* norm — supplied the vocabulary. Muon itself appeared
in late 2024 as a practical recipe validated on a speedrun benchmark, not as a
theory paper.

**2. From recipe to geometry (early–mid 2025).** Work in this period asked what
Muon *is*. Answers arrived as norm-constrained steepest descent, a linear
minimization oracle over a spectral-norm ball, a non-Euclidean trust region, and a
Lion-K instance implying an implicit spectral-norm constraint. These framings are
related but not identical, and they were largely derived for an idealized Muon with
exact orthogonalization.

**3. Scaling evidence (2025).** Two results moved the field: a production MoE
trained with a weight-decayed, RMS-matched Muon, and a study arguing Muon retains
data efficiency past the critical batch size. Both are best read carefully — the
first reports FLOPs, the second a compute-versus-wall-clock frontier, and neither
ran an independent per-optimizer sweep at every scale.

**4. Faster orthogonalization (2025).** Once Muon mattered, the Newton–Schulz inner
solver became a target: minimax-optimal iteration-varying polynomials, Chebyshev-type
coefficient families, and preconditioned initial guesses. A quieter and more
important finding emerged alongside — the number of iterations interacts with the
learning rate and momentum, so the "approximation" is a hyperparameter, not an
implementation detail.

**5. Distributed and low-memory variants (2025–2026).** Muon's whole-matrix
operation conflicts with parameter sharding. Solutions include low-rank momentum
with error feedback, block-periodic and tiled orthogonalization, random submatrix
sampling, sign-compressed aggregation, and deliberately unbalanced sharding that
gives one rank full ownership of each matrix. In parallel, 8-bit and 4-bit
optimizer states were shown to be workable.

**6. How much flattening? (2026).** The central research question changed. Rather
than asking whether orthogonalizing momentum helps, papers began asking what
spectral transformation to apply and by how much: fractional powers `σ → σ^p`,
singular-value clipping, nuclear-norm shrinkage, curvature-weighted geometries,
row- and neuron-level normalization, second-moment preconditioning around the
orthogonalization step, and hybrid Muon/Adam designs. A theoretical result under an
isotropic-curvature model puts this plainly: flattening is directionally right, but
*full* flattening is optimal only past a curvature phase transition.

**7. Critical results (2025–2026).** Under equal-density tuning, matrix methods'
advantage over AdamW shrinks as models grow. Muon provably does not converge on
convex Lipschitz functions, and the error-feedback fix that repairs the theory makes
training worse. An ablation replacing the singular values with random noise matched
Muon at small scale. On controlled low-rank matrix factorization, Muon did not
consistently beat a tuned AdamW. Late-training pathologies — overshoot near the
solution, post-grokking collapse — have been documented in small models.

**8. Beyond pretraining (2026).** Attention moved to fine-tuning, post-training,
and other domains. Switching an Adam-pretrained checkpoint to Muon frequently hurts
unless the update magnitude is constrained, for instance through LoRA. Results
outside language-model pretraining — vision transformers, diffusion transformers,
state-space models, recommendation, tabular data, scientific ML, RL — are mostly
positive but frequently *localized*: to specific layers, specific training recipes,
or specific spectral regimes.

## Suggested Reading Paths

These are entry points, not rankings. Each item is chosen for what it teaches, not
for prestige or citation count.

**New to Muon**

1. [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — the algorithm from its author, with the original motivation.
2. [Deriving Muon](https://jeremybernste.in/writing/deriving-muon) — the shortest path from "why a norm" to "why Newton–Schulz".
3. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — the two changes that made it work at production scale.
4. [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046) — read immediately afterwards, to calibrate expectations.
5. [What Really Matters in Matrix-Whitening Optimizers?](https://arxiv.org/abs/2510.25000) — which component of the family is actually doing the work.

**Theory and geometry**

1. [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325) — the norm-selection framing everything else builds on.
2. [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529) — the LMO formulation, with Muon as a special case.
3. [Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization](https://arxiv.org/abs/2503.12645) — the trust-region reading and its rates.
4. [Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](https://arxiv.org/abs/2502.04664) — what solution the method is biased toward.
5. [Muon Does Not Converge on Convex Lipschitz Functions](https://arxiv.org/abs/2605.08980) — the sharpest boundary on what the theory can claim.
6. [Isotropic Curvature Model for Understanding Deep Learning Optimization](https://arxiv.org/abs/2511.00674) — whether full orthogonalization is optimal at all.

**Large-scale and distributed training**

1. [Dion: Distributed Orthonormalized Updates](https://arxiv.org/abs/2504.05295) — the canonical statement of the sharding problem.
2. [MatrixFSDP: communication-free matrix optimizers under ZeRO-3 parameter sharding](https://arxiv.org/abs/2607.05895) — ownership-based placement that removes the optimizer-step collective.
3. [MuonBP: Faster Muon via Block-Periodic Orthogonalization](https://arxiv.org/abs/2510.16981) — trading exactness for communication under tensor parallelism.
4. [SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales](https://arxiv.org/abs/2607.20548) — large-batch behaviour with a production framework integration.
5. [Optimal Scaling Needs Optimal Norm](https://arxiv.org/abs/2510.03871) — how learning rate and batch size should scale together.

**Practical implementation**

1. [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — the reference code; read the parameter-group split first.
2. [The Polar Express](https://arxiv.org/abs/2505.16932) — the current best-understood inner solver.
3. [Beyond the Ideal: Analyzing the Inexact Muon Update](https://arxiv.org/abs/2510.19933) — why the iteration count must be tuned with the learning rate.
4. [Effective Quantization of Muon Optimizer States](https://arxiv.org/abs/2509.23106) — what the optimizer state costs and how far it compresses.
5. [Benchmarking Optimizers for Large Language Model Pretraining](https://arxiv.org/abs/2509.01440) — weight decay on 2D parameters, batch-size sensitivity, and schedule choices.

**Fine-tuning and applications**

1. [Can Muon Fine-tune Adam-Pretrained Models?](https://arxiv.org/abs/2605.10468) — the clearest statement of the optimizer-transfer problem.
2. [Rethinking Muon Beyond Pretraining](https://arxiv.org/abs/2605.19282) — where flattening every singular value actively hurts.
3. [LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold](https://arxiv.org/abs/2606.12921) — the spectral rule rederived for factored updates.
4. [Muon in Vision Transformers](https://arxiv.org/abs/2605.24770) — how much of the benefit is the optimizer versus the training recipe.
5. [When Does Muon Help Agentic Reinforcement Learning?](https://arxiv.org/abs/2607.16169) — a cautious read on post-training transfer.

**Limitations and critical evidence**

1. [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046) — the tuning-parity critique, with the advantage decaying as scale grows.
2. [Muon is Not That Special: Random or Inverted Spectra Work Just as Well](https://arxiv.org/abs/2605.11181) — a direct challenge to the geometry explanation.
3. [Reassessing Muon for Matrix Factorization](https://arxiv.org/abs/2607.13246) — the advantage does not reproduce on a controlled problem.
4. [Towards Understanding the Power and Limits of the Muon Optimizer](https://arxiv.org/abs/2606.21514) — why an early-training advantage need not persist.
5. [On MUON optimization: from non-convergence to an error analysis](https://arxiv.org/abs/2608.04607) — a theorem-grade non-convergence result.
6. [To Use or not to Use Muon: How Simplicity Bias in Optimizers Matters](https://arxiv.org/abs/2603.00742) — what is lost when the learning order is flattened.

## Recent Additions

Papers first publicly released in roughly the trailing ten weeks. This section is
capped at 15 entries and is pruned as it fills; every paper here also has a
permanent entry in its topical section below.

- **[Second-Order Muon Done Right: A Principled Marriage of Spectral Geometry and Curvature](https://arxiv.org/abs/2608.09763)** — 2026-08-10
- **[Post-Grokking Collapse at the Representation-Readout Interface in Muon-Trained Transformers](https://arxiv.org/abs/2608.07436)** — 2026-08-07
- **[Muon on the Stiefel Manifold Admits an Exact Closed-Form Update](https://arxiv.org/abs/2608.06218)** — 2026-08-06
- **[MALT: Lightweight Curvature-Aware Muon via Diagonal Preconditioning](https://arxiv.org/abs/2608.05088)** — 2026-08-05
- **[On MUON optimization: From non-convergence to an error analysis with Polar Express and the Newton-Schulz polynomial from implementations](https://arxiv.org/abs/2608.04607)** — 2026-08-05
- **[Muon Meets Mamba: Spectral Optimization for State Space Models](https://arxiv.org/abs/2608.03941)** — 2026-08-04
- **[CMuon: Accelerating and Stabilizing Diffusion Transformer Training via Chunked Momentum Orthogonalization](https://arxiv.org/abs/2608.02502)** — 2026-08-03, ECCV 2026
- **[Sharpness-Aware Minimization and Muon: Robustness under the Spectral Norm](https://arxiv.org/abs/2607.26001)** — 2026-07-28
- **[Scale Weight Decay and Train Better](https://arxiv.org/abs/2607.23777)** — 2026-07-26
- **[An Isotropy-Preserving Spectral Cap for Muon: Theory and Three Case Studies](https://arxiv.org/abs/2607.19771)** — 2026-07-22
- **[When Does Muon Help Agentic Reinforcement Learning?](https://arxiv.org/abs/2607.16169)** — 2026-07-17
- **[Reassessing Muon for Matrix Factorization](https://arxiv.org/abs/2607.13246)** — 2026-07-14
- **[SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales](https://arxiv.org/abs/2607.20548)** — 2026-07-13
- **[MatrixFSDP: communication-free matrix optimizers under ZeRO-3 parameter sharding](https://arxiv.org/abs/2607.05895)** — 2026-07-07
- **[The Active Ingredient in Muon's Grokking](https://arxiv.org/abs/2607.20512)** — 2026-07-06

## Papers by Topic

Each paper appears in exactly one section, sorted by first public submission date,
newest first — except the foundational subsection, which reads forward in time.
Author lists longer than three names are abbreviated with "et al.". Dates in the
entries are years; exact first-public dates are used for ordering and appear in the
[timeline](docs/research-landscape.md#chronological-timeline).

### Origins and Foundational Perspectives

Listed oldest first, so the lineage reads forward.

- **[Preconditioned Spectral Descent for Deep Learning](https://papers.nips.cc/paper/5795-preconditioned-spectral-descent-for-deep-learning)** — David E. Carlson, Edo Collins, Ya-Ping Hsieh, et al., NIPS 2015. Argues that steepest descent under Schatten-∞ (spectral) geometry gives tighter progress bounds than Frobenius geometry for certain model classes, and confronts the cost of the spectral step — the clearest prior art for treating spectral geometry as an optimizer design choice.
- **[The Geometry of Sign Gradient Descent](https://arxiv.org/abs/2002.08056)** — Lukas Balles, Fabian Pedregosa, Nicolas Le Roux, Preprint, 2020. Develops steepest descent under a norm and norm-smoothness for sign-based methods, isolating the axis-aligned curvature conditions that favour them — the coordinate-dependent foil against which rotation-invariant spectral geometry is later argued.
- **[Orthogonalising gradients to speed up neural network optimisation](https://arxiv.org/abs/2202.07052)** — Mark Tuddenham, Adam Prügel-Bennett, Jonathan Hare, Preprint, 2022. Inserts an orthogonalization step before the SGD update to diversify learned representations, reporting reduced training time on ImageNet and CIFAR-10 — the closest direct predecessor to Muon's update, later analyzed as a special case by trust-region work.
- **[Scalable Optimization in the Modular Norm](https://arxiv.org/abs/2405.14813)** — Tim Large, Yang Liu, Minyoung Huh, et al., NeurIPS 2024. Defines a norm on a network's whole weight space built recursively alongside the architecture, so learning rates transfer across width and depth without optimizer-specific rescaling. [Code](https://github.com/modula-systems/modula)
- **[Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)** — Jeremy Bernstein, Laker Newhouse, OPT 2024 workshop at NeurIPS. Reframes Adam, Shampoo, and Prodigy as steepest descent under particular norms once exponential moving averages are switched off, splitting optimizer design into a per-tensor norm choice and a step-size choice; the equivalences hold under idealizations and the paper offers no experiments.
- **[Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265)** — Jeremy Bernstein, Laker Newhouse, ICML 2025, 2024. Derives layer-wise duality maps from operator norms, providing the formal scaffolding that Muon's update instantiates for linear layers.
- **[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)** — Keller Jordan, first-party technical post, 2024. The canonical primary source: SGD-momentum on matrix-shaped gradients followed by Newton–Schulz orthogonalization, introduced through NanoGPT speedrun records. There is no corresponding paper. [Code](https://github.com/KellerJordan/Muon)
- **[Duality, Weight Decay, and Metrized Deep Learning](https://www.lakernewhouse.com/thesis.pdf)** — Laker Newhouse, MIT MEng thesis, 2025. Develops the duality-map view of gradient optimization together with practical methods for enforcing Lipschitz bounds on weights, covering Muon and weight-constraint techniques for stable transformer training.

### Theory, Convergence, and Implicit Bias

- **[Muon on the Stiefel Manifold Admits an Exact Closed-Form Update](https://arxiv.org/abs/2608.06218)** — Mikhail Solonko, Alexander Molozhavenko, Maxim Rakhuba, Preprint, 2026. Shows that restricting the Muon subproblem to matrices with orthonormal columns is exactly solvable rather than requiring the heuristic or iterative schemes used by earlier Stiefel extensions, and builds an algorithm plus first-order convergence theory on that result.
- **[Muon as a Residual Connection](https://arxiv.org/abs/2607.01124)** — Hao Huang, Preprint, 2026. Offers a mechanistic reading in which orthogonalized updates behave like an implicit residual connection, trading immediate fit of the local objective for representations downstream layers exploit more easily; the evidence is controlled two-layer linear experiments and the author states it is not a systematic study. [Code](https://github.com/huanghao-sss/muon_interpretation)
- **[Muon learns balanced solutions in matrix factorization without slow saddle-to-saddle dynamics](https://arxiv.org/abs/2606.30509)** — Mark Rhee, Jamie Simon, Dhruva Karkada, Preprint, 2026. Characterizes how Muon's trajectories in matrix factorization differ from gradient descent — uniform mode learning instead of staircase saddle escapes, and stability above the sharpness-set learning-rate threshold — while noting that a constant learning rate oscillates indefinitely around the solution manifold. [Code](https://github.com/dkarkada/muon-mfac)
- **[Free Heavy-Tailed Lunch for Muon: A Theoretical Justification of Empirical Success](https://arxiv.org/abs/2606.14560)** — Florian Hübler, Thomas Pethick, Suvrit Sra, Preprint, 2026. Proves that under heavy-tailed gradient noise with bounded p-th moments, Muon-style non-Euclidean methods reach optimal sample complexity with no dimension penalty, and supplies a matching first-order lower bound. [Code](https://github.com/fhueb/ht-schatten-experiments)
- **[The Spectral Dynamics and Noise Geometry of Muon](https://arxiv.org/abs/2606.08388)** — Pierfrancesco Beneventano, Mahmoud Abdelmoneum, Tomaso Poggio, Preprint, 2026. Argues the polar-factor update imposes a flat-spectrum, maximum-entropy bias rather than the low-rank or nuclear-norm bias often attributed to it, and reports that the resulting benefit is regime-dependent — a small vision control reverses the optimizer ranking observed on NanoGPT.
- **[Why Muon Outperforms Adam: A Curvature Perspective](https://arxiv.org/abs/2606.04662)** — Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, et al., Preprint, 2026. Attributes the observed efficiency gap to a smaller second-order curvature penalty rather than any first-order gain, isolating normalized directional sharpness as the responsible quantity; the analysis is empirical at 124M scale and its one theorem compares against gradient descent rather than Adam.
- **[Spectral Flattening Is All Muon Needs: How Orthogonalization Controls Learning Rate and Convergence](https://arxiv.org/abs/2605.13079)** — Tien-Phat Nguyen, Truong Nguyen, Minh-Phuc Truong, et al., Preprint, 2026. Explains Muon's unusually large usable step size by showing the maximum stable learning rate is governed by the average rather than the maximum gradient singular value, and recasts the method as preconditioned gradient descent under Kronecker-factored curvature.
- **[Dimension-Free Saddle-Point Escape in Muon](https://arxiv.org/abs/2605.09331)** — Yanlin Long, Yufei Gu, Zeke Xie, Preprint, 2026. Uses matrix perturbation theory to argue Muon leaves flat saddles in time independent of parameter dimension where element-wise adaptive methods incur a dimensional penalty; the guarantee requires a low-rank signal structure and an eventual spectral gap, without which it is vacuous.
- **[Muon Dynamics as a Spectral Wasserstein Flow](https://arxiv.org/abs/2604.04891)** — Gabriel Peyré, Preprint, 2026. Constructs an optimal-transport framework in which spectrally normalized mean-field training is a gradient flow, with a family of distances where the trace norm recovers standard W2 and the operator norm recovers Muon's geometry; the setting is deterministic, continuous-time, and infinite-width. [Code](https://github.com/gpeyre/muon-dynamics)
- **[Sharp Capacity Scaling of Spectral Optimizers in Learning Associative Memory](https://arxiv.org/abs/2603.26554)** — Juno Kim, Eshaan Nichani, Denny Wu, et al., Preprint, 2026. Derives sharp one-step recovery rates showing Muon stores substantially more associations than SGD and tolerates a much larger critical batch size, while noting that in the multi-step analysis SGD eventually reaches the same information-theoretic limit.
- **[Muon Converges under Heavy-Tailed Noise: Nonconvex Hölder-Smooth Empirical Risk Minimization](https://arxiv.org/abs/2603.15059)** — Hideaki Iiduka, Preprint, 2026. Proves almost-sure stationarity and rate bounds when gradient noise has only a bounded p-th moment, and shows Muon's step-size and batch-size conditions are strictly weaker than mini-batch SGD's; the analysis assumes exact orthogonalization and no momentum.
- **[The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](https://arxiv.org/abs/2602.16340)** — Eitan Gronich, Gal Vardi, Preprint, 2026. Extends implicit-bias analysis to momentum steepest descent, showing Muon, Signum, Adam, and hybrids converge to KKT points of margin-maximization problems posed in *different* norms.
- **[Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](https://arxiv.org/abs/2602.05725)** — Binghui Li, Kaifei Wang, Han Zhong, et al., ICML 2026. Uses an associative-memory model with a hierarchical frequency distribution to explain why Muon spreads progress evenly across frequency components where gradient descent stalls on rare ones; the separation is derived in a stylized setting with block-symmetric gradient structure.
- **[Improved Convergence Rates of Muon Optimizer for Nonconvex Optimization](https://arxiv.org/abs/2601.19400)** — Shuntaro Nagashima, Hideaki Iiduka, Preprint, 2026. Tightens existing nonconvex guarantees using a direct argument that drops structural assumptions on the update rule required by earlier analyses; no experiments accompany the bound.
- **[Preconditioning Benefits of Spectral Orthogonalization in Muon](https://arxiv.org/abs/2601.13474)** — Jianhao Ma, Yu Huang, Yuejie Chi, et al., Preprint, 2026. Proves on matrix factorization and linear-transformer in-context learning that orthogonalization decouples the dynamics into independent scalar sequences, which is why iteration count stops depending on conditioning; the analysis covers a simplified Muon and excludes per-step orthogonalization cost.
- **[When do spectral gradient updates help in deep learning?](https://arxiv.org/abs/2512.04299)** — Damek Davis, Dmitriy Drusvyatskiy, Preprint, 2025. Gives a layerwise criterion comparing a gradient nuclear-to-Frobenius ratio against the stable rank of activations, validated at NanoGPT scale; the guarantee is essentially one-step and concerns the canonical spectral update rather than Muon's practical details.
- **[An Exploration of Non-Euclidean Gradient Descent: Muon and its Many Variants](https://arxiv.org/abs/2510.09827)** — Michael Crawshaw, Chirag Modi, Mingrui Liu, et al., Preprint, 2025. Unifies the growing variant space as constrained versus regularized steepest descent under product norms, and proposes a variant with markedly wider learning-rate robustness.
- **[Muon Outperforms Adam in Tail-End Associative Memory Learning](https://arxiv.org/abs/2509.26030)** — Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, et al., ICLR 2026, 2025. Localizes the advantage to associative-memory parameters — value/output attention weights and FFNs — arguing the spectral update learns rare tail classes that Adam under-fits on heavy-tailed data; the theory assumes orthonormal embeddings and disables momentum.
- **[Muon Optimizes Under Spectral Norm Constraints](https://arxiv.org/abs/2506.15054)** — Lizhang Chen, Jonathan Li, Qiang Liu, OPT 2025 workshop at NeurIPS. Shows Muon is the Lion-K optimizer instantiated with the nuclear norm, from which it follows that Muon with decoupled weight decay implicitly solves a spectral-norm-constrained problem.
- **[Lions and Muons: Optimization via Stochastic Frank-Wolfe](https://arxiv.org/abs/2506.04192)** — Maria-Eleni Sfyraki, Jun-Kun Wang, Preprint, 2025. Recasts Lion and weight-decayed Muon as instances of stochastic Frank–Wolfe, then builds variants robust to heavy-tailed noise.
- **[On the Convergence Analysis of Muon](https://arxiv.org/abs/2505.23737)** — Wei Shen, Ruichuan Huang, Minhui Huang, et al., Preprint, 2025. Derives deterministic and stochastic rates and identifies the Hessian conditions — low-rank and approximately block-diagonal — under which Muon provably beats gradient descent; the analysis assumes exact SVD where practice uses Newton–Schulz.
- **[From Muon to Gluon: Bridging Theory and Practice of LMO-based Optimizers for LLMs](https://arxiv.org/abs/2505.13416)** — Artem Riabinin, Egor Shulgin, Kaja Gruntkowska, et al., ICML 2026, 2025. Introduces a layer-wise generalized-smoothness model whose theoretically prescribed stepsizes match tuned practice, closing a gap earlier analyses left open. (Titled "Gluon: Making Muon & Scion Great Again!" on arXiv.)
- **[Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization](https://arxiv.org/abs/2503.12645)** — Dmitry Kovalev, Preprint, 2025. Recasts orthogonalized gradient descent as a first-order trust-region method under the spectral norm and builds a stochastic non-Euclidean trust-region method with momentum recovering Muon, normalized SGD, and signSGD-with-momentum as special cases; there are no experiments.
- **[Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529)** — Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, et al., ICML 2025. Builds a stochastic optimizer family around the linear minimization oracle over a norm ball, unifying Muon with several existing methods and yielding Scion, whose explicit norm choice gives hyperparameter transfer across scale. [Code](https://github.com/LIONS-EPFL/scion)
- **[Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](https://arxiv.org/abs/2502.04664)** — Chen Fan, Mark Schmidt, Christos Thrampoulidis, NeurIPS 2025 (Spotlight). Proves normalized steepest descent and its momentum variant converge to norm-specific max-margin solutions on multiclass separable data, with spectral descent and Muon converging to the spectral-norm max-margin solution.
- **[A Note on the Convergence of Muon](https://arxiv.org/abs/2502.02900)** — Jiaxiang Li, Mingyi Hong, Preprint, 2025. A short note giving convergence guarantees for the heavy-ball form of Muon and a spectral-norm variant, framing the update as minimizing a quadratic model under the spectral norm; batch-free convergence for the spectral variant is left open.

### Scaling, Parameterization, and Critical Batch Size

- **[SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales](https://arxiv.org/abs/2607.20548)** — Mikail Khona, Aditya Vavre, Boxiang Wang, et al., Preprint, 2026. A production-framework comparison of higher-order optimizers sweeping batch sizes to roughly 100M tokens, which fixes SOAP's large-batch instability and ships a layer-wise distributed optimizer for Megatron-LM. [Code](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)
- **[Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899)** — Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, et al., Preprint, 2026. Argues Muon's advantage erodes at scale because of how decoupled weight decay controls weight norms, and replaces that mechanism with a per-matrix Frobenius-norm constraint, reporting token-equivalent gains and tighter learning-rate transfer; the Frobenius surrogate only approximates spectral control.
- **[Rethinking Language Model Scaling under Transferable Hypersphere Optimization](https://arxiv.org/abs/2603.28743)** — Liliang Ren, Yang Liu, Yelong Shen, et al., Preprint, 2026. Introduces a parameterization under a Frobenius-sphere constraint that makes the optimal Muon learning rate transfer across width, depth, token budget, and mixture-of-experts granularity from a single small-scale tuning run.
- **[Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales for Stochastic Sign and Spectral Descent](https://arxiv.org/abs/2602.03001)** — Hiroki Naganuma, Shagun Gupta, Youssef Briki, et al., Preprint, 2026. Derives gradient noise scales in the dual norms native to sign-based and spectral optimizers rather than reusing the Euclidean form, and turns them into an adaptive batch-size schedule with a distributed variance estimator; the reported savings are in optimizer steps at a single 160M scale.
- **[Controlled LLM Training on Spectral Sphere](https://arxiv.org/abs/2601.08393)** — Tian Xie, Haoming Luo, Haoyu Tang, et al., ICML 2026 (Oral), 2026. Enforces module-wise spectral constraints on both weights and updates in a way that aligns with maximal-update parameterization, implemented in Megatron for dense and mixture-of-experts models.
- **[Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales](https://openreview.net/forum?id=Ei6IsmxYrb)** — Shikai Qiu, Zixi Chen, Hoang Phan, et al., NeurIPS 2025. Shows the reported advantage of matrix-preconditioned optimizers over AdamW largely survives scaling only when hyperparameters are transferred with the correct scaling rules, and reports a consistent compute-matched speedup for Muon, SOAP, and Shampoo once they are; the study tops out at 1.4B. [Code](https://github.com/charliezchen/scaling-matrix-preconditioning)
- **[Optimal Scaling Needs Optimal Norm](https://arxiv.org/abs/2510.03871)** — Oleg Filatov, Jiangtao Wang, Jan Ebert, et al., Preprint, 2025. Finds that the jointly optimal learning rate and batch size across model and data scales are pinned to a single invariant — the operator norm of the output layer — across more than two thousand runs; the authors state the condition is necessary but not sufficient. [Code](https://github.com/SDLAML/disco)
- **[Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)** — Kimi Team, technical report, 2025. Introduces MuonClip, pairing Muon with QK-clipping, and reports pretraining at 15.5T tokens without loss spikes — the largest-scale published datapoint for Muon-family training, though as a system report rather than a controlled comparison.
- **[Convergence Bound and Critical Batch Size of Muon Optimizer](https://arxiv.org/abs/2507.01598)** — Naoki Sato, Hiroki Naganuma, Hideaki Iiduka, Preprint, 2025. Proves convergence for four practical Muon variants — with and without Nesterov momentum and weight decay — and derives the critical batch size minimizing total computational cost; the single-matrix analysis does not model layer-wise gradient heterogeneity. (Retitled from an earlier version; cite the current title.)
- **[Practical Efficiency of Muon for Pretraining](https://arxiv.org/abs/2505.02222)** — Ishaan Shah, Anthony M. Polloreno, Karl Stratos, et al., Preprint, 2025. Argues Muon extends the compute-versus-wall-clock frontier relative to AdamW by retaining data efficiency at batch sizes well past the critical batch size, and makes its hyperparameters transferable by combining muP with a multi-scale grid-refinement procedure; results reach 4B parameters.
- **[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)** — Jingyuan Liu, Jianlin Su, Xingcheng Yao, et al., Preprint, 2025. Shows that adding weight decay and rescaling the update so per-matrix update RMS matches AdamW's makes Muon work at production scale, demonstrated on a 16B-total mixture-of-experts trained on 5.7T tokens; the headline efficiency claim is measured in FLOPs, and AdamW's tuned hyperparameters were reused for Muon rather than re-swept. [Code](https://github.com/MoonshotAI/Moonlight)

### Spectral Shaping, Adaptivity, and Muon Variants

- **[Second-Order Muon Done Right: A Principled Marriage of Spectral Geometry and Curvature](https://arxiv.org/abs/2608.09763)** — Tong Che, Preprint, 2026. Observes that the polar update solves the *unweighted* spectral oracle exactly, and generalizes it to a data-dependent weighted geometry whose raw update remains exact regardless of how the weighting maps were estimated, with the maps refreshed lazily to amortize cost.
- **[MALT: Lightweight Curvature-Aware Muon via Diagonal Preconditioning](https://arxiv.org/abs/2608.05088)** — Tongle Wu, Huanyu Dong, Ying Sun, et al., Preprint, 2026. Adds row-wise and column-wise squared-gradient statistics as two diagonal preconditioners that conjugate the momentum around the Newton–Schulz step, restoring scale afterwards by Frobenius grafting; evidence is confined to GPT-2-scale pretraining.
- **[An Isotropy-Preserving Spectral Cap for Muon: Theory and Three Case Studies](https://arxiv.org/abs/2607.19771)** — Jiachun Li, Preprint, 2026. Argues Muon removes an implicit norm-growth brake that SGD retains under loss scale-invariance, and adds a cheap cap limiting growth of the top singular direction only; self-described as a preliminary report.
- **[Aurora: A Leverage-Aware Spectral Optimizer](https://arxiv.org/abs/2606.27715)** — Alec Dewulf, Dhruv Pai, Li Yang, et al., Preprint, 2026. Identifies that on tall matrices the update's row norms can be badly unequal, starving individual neurons, and equalizes them by solving semi-orthogonality and equal-row-norm jointly so the update does not drift off the polar factor; applies to tall matrices only, with square and wide parameters falling back to standard Muon. [Code](https://github.com/tilde-research/aurora-release)
- **[Tensorion: A Tensor-Aware Generalization of the Muon Optimizer](https://arxiv.org/abs/2606.25975)** — Vladimir Bogachev, Vladimir Aletov, Alexander Molozhavenko, et al., Preprint, 2026. Lifts the norm-constrained update from matrices to higher-order tensors by choosing a tensor norm whose linear minimization oracle stays computable through adaptively selected unfoldings, collapsing exactly to Muon in the matrix case; evidence is computer vision only. [Code](https://github.com/MTML-LAB/Tensorion)
- **[PowerMuon: Muon with Fractional Spectral Powers](https://arxiv.org/abs/2606.13867)** — Yihe Dong, Will Sawin, COLM 2026, 2026. Flattens the spectrum only partially by sending each singular value to `σ^p` for rational `p` in (0,1), and — after proving no fixed univariate polynomial iteration can realize such a power — supplies bivariate recurrences so the update still costs only matrix multiplications; the authors report it is *worse* than Muon when pretraining from random initialization, with gains specific to fine-tuning. [Code](https://github.com/princeton-pli/muon-p)
- **[OptMuon: Closed-Loop Orthogonalized Momentum Methods for Stochastic Optimization with Zero-Noise Optimality](https://arxiv.org/abs/2606.08783)** — Ganzhao Yuan, Preprint, 2026. Keeps the polar direction but replaces the externally scheduled step magnitude with an AdaGrad-Norm-style scalar computed from the realized trajectory, giving rates that collapse to the noiseless case without retuning; the analysis presumes exact polar factorization and has no empirical component.
- **[MONA: Muon Optimizer with Nesterov Acceleration for Scalable Language Model Training](https://arxiv.org/abs/2605.26842)** — Jiacheng Li, Jianchao Tan, Hongtao Xu, et al., Preprint, 2026. Injects an extrapolation term built from an EMA of successive gradient differences into the gradient before the momentum buffer, so the buffer being orthogonalized already carries a finite-difference curvature proxy; evaluated on mixture-of-experts models to 68B, with all optimizers sharing one learning rate rather than per-optimizer sweeps.
- **[MuCon: Clipped Muon Updates for LLM Training](https://arxiv.org/abs/2605.26459)** — Albert Yi, Preprint, 2026. Replaces full flattening with a cap that limits each singular value at a threshold while leaving smaller ones untouched, then analyzes whether matrix-multiplication-style iterations can implement the cap; the conclusion is cautionary — conditioning degrades when many singular values sit near the threshold, and there are no training experiments.
- **[Anytime Training with Schedule-Free Spectral Optimization](https://arxiv.org/abs/2605.23061)** — Anuj Apte, Pranav Deshpande, Niraj Kumar, et al., Preprint, 2026. Combines schedule-free iterate averaging with a row-normalized spectral update and identifies that applying weight decay at the fast iterate rather than the averaged one is what keeps long-horizon runs stable, yielding usable checkpoints without committing to a training length.
- **[AMUSE: Anytime Muon with Stable Gradient Evaluation](https://arxiv.org/abs/2605.22432)** — Jueun Kim, Baekrok Shin, Jihun Yun, et al., Preprint, 2026. Changes where the gradient is evaluated rather than how the update is orthogonalized, moving the evaluation point from the fast iterate toward a schedule-free average over training to damp oscillation across sharp valley walls.
- **[Pion: A Spectrum-Preserving Optimizer via Orthogonal Equivalence Transformation](https://arxiv.org/abs/2605.12492)** — Kexuan Shi, Hanxuan Li, Zeju Qiu, et al., Preprint, 2026. Departs from additive updates entirely, applying left and right orthogonal transformations so the weight matrix's full singular-value spectrum is held fixed throughout training rather than reshaped. (Distinct from the similarly named method in [2605.19282](https://arxiv.org/abs/2605.19282).)
- **[Muown: Row-Norm Control for Muon Optimization](https://arxiv.org/abs/2605.10797)** — Kai Lion, Florian Hübler, Bingcong Li, et al., Preprint, 2026. Traces Muon's weight-decay sensitivity to drift in per-row magnitudes and promotes row magnitude to an explicit optimizer state updated under ℓ∞ geometry, leaving the orthogonalized direction untouched; evaluated from 124M to 2.7B.
- **[Intrinsic Muon: Spectral Optimization on Riemannian Matrix Manifolds](https://arxiv.org/abs/2605.09238)** — Yibang Li, Bihari Lal Pandey, Ravi Sah, et al., Preprint, 2026. Lifts any unitarily invariant Euclidean norm to an intrinsic tangent-space norm under a Riemannian metric, giving closed-form Muon-style updates on the fixed-rank, SPD, Stiefel, and Grassmann manifolds.
- **[Muon²: Boosting Muon via Adaptive Second-Moment Preconditioning](https://arxiv.org/abs/2604.09967)** — Ziyue Liu, Ruijie Zhang, Zhengyang Wang, et al., Preprint, 2026. Applies Adam-style element-wise second-moment scaling to the momentum before orthogonalization, which both improves the update and conditions the input well enough to cut Newton–Schulz iterations substantially; carries Adam's full second-moment buffer as extra state.
- **[The Newton-Muon Optimizer](https://arxiv.org/abs/2604.01472)** — Zhehang Du, Weijie Su, Preprint, 2026. Derives Muon from a quadratic surrogate built from the gradient, an output-space curvature matrix, and the input data matrix, showing standard Muon is the case that drops right-preconditioning, and adds that term back by scaling with inverse input second moments; the reported gains are single-digit percentages on GPT-2.
- **[MuonEq: Balancing Before Orthogonalization with Lightweight Equilibration](https://arxiv.org/abs/2603.28254)** — Da Chang, Qiankun Shi, Lvgang Zhang, et al., Preprint, 2026. Inserts a cheap row and column rescaling of the momentum *before* the fixed-step Newton–Schulz iteration, on the argument that finite-step orthogonalization quality is governed by the input's conditioning; the benefit is contingent on the non-converged iteration regime.
- **[Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning](https://arxiv.org/abs/2603.09697)** — Yechen Zhang, Shuhao Xing, Junhao Huang, et al., Preprint, 2026. Argues the uniform spectral step implicitly assumes an isotropic landscape and runs the spectral step inside a whitened frame built from Kronecker-factored curvature statistics, reporting roughly 12% fewer steps at 160M–800M scale.
- **[NuMuon: Nuclear-Norm-Constrained Muon for Compressible LLM Training](https://arxiv.org/abs/2603.03597)** — Hadi Mohaghegh Dolatabadi, Thalaiyasingam Ajanthan, Sameera Ramasinghe, et al., Preprint, 2026. Documents that Muon's full-rank updates nonetheless yield low-rank trained weights, then constrains the update's nuclear norm — shrinking singular values rather than flattening them — to push weights further toward low rank for downstream compression.
- **[MUON+: Towards More Effective Muon via One Additional Normalization Step for LLM Pre-training](https://arxiv.org/abs/2602.21545)** — Ruijie Zhang, Yequan Zhao, Ziyue Liu, et al., Preprint, 2026. Shows Newton–Schulz steps do not remove row and column norm imbalance in the update and can worsen it, then patches this with a single stateless normalization applied after orthogonalization; the title, author list, and experimental scale all changed materially between versions.
- **[Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized Momentum](https://arxiv.org/abs/2602.17080)** — Minxin Zhang, Yuxuan Liu, Hayden Schaeffer, Preprint, 2026. Adds scalar (orthogonality-preserving) and diagonal neuron-wise adaptive scaling on top of the orthogonalized momentum, evaluated on GPT-2 pretraining.
- **[Delving into Muon and Beyond: Deep Analysis and Extensions](https://arxiv.org/abs/2602.04669)** — Xianbiao Qi, Marco Chen, Jiaquan Ye, et al., Preprint, 2026. Recasts Muon as the `p = 0` member of a family of spectral maps `U Σ^p Vᵀ`, adds intermediate powers computable without explicit SVD, and argues from controlled comparisons that Muon acts as a spectral *normalizer* rather than a strictly better optimizer — reporting that it underperforms Adam once applied to second-moment-normalized updates. [Code](https://github.com/Ocram7/BeyondMuon)
- **[Mano: Restriking Manifold Optimization for LLM Training](https://arxiv.org/abs/2601.23000)** — Yufei Gu, Zeke Xie, Preprint, 2026. Projects momentum onto the tangent space and constrains it to a rotational Oblique manifold, reporting improvements over both AdamW and Muon with less memory and compute.
- **[Manifold constrained steepest descent](https://arxiv.org/abs/2601.21487)** — Kaiwei Yang, Lexiao Lai, Preprint, 2026. Gives a single-loop manifold linear-minimization-oracle scheme, avoiding the nested tangent-space solves earlier manifold formulations require, specialized to the Stiefel manifold.
- **[Variance-Adaptive Muon](https://arxiv.org/abs/2601.14603)** — Jingru Li, Yibo Fan, Huan Li, Preprint, 2026. Applies variance-scaled normalization to the momentum *before* orthogonalization rather than after, reporting a speedup on LLaMA-scale pretraining.
- **[Isotropic Curvature Model for Understanding Deep Learning Optimization: Is Gradient Orthogonalization Optimal?](https://arxiv.org/abs/2511.00674)** — Weijie Su, Preprint, 2025. Shows that under an isotropic-curvature convex model the optimal update flattens singular values, but full orthogonalization becomes optimal only past a curvature phase transition — making Muon directionally right without being strictly optimal, and framing the partial-flattening question precisely.
- **[MARS-M: When Variance Reduction Meets Matrices](https://arxiv.org/abs/2510.21800)** — Yifeng Liu, Angela Yuan, Quanquan Gu, Preprint, 2025. Combines MARS-style variance reduction with Muon's matrix preconditioning, improving the theoretical rate and evaluating on language modelling and vision. [Code](https://github.com/AGI-Arena/MARS)
- **[NorMuon: Making Muon more efficient and scalable](https://arxiv.org/abs/2510.05491)** — Zichong Li, Liming Liu, Chen Liang, et al., Preprint, 2025. Observes that orthogonalized updates still have highly non-uniform per-neuron row norms and adds per-neuron second moments with row-wise normalization after orthogonalization, shipped with an FSDP2-compatible implementation; the reported gains are in training steps against baselines that were not tuned to the same density. [Code](https://github.com/zichongli5/NorMuon)
- **[Drop-Muon: Update Less, Converge Faster](https://arxiv.org/abs/2510.02239)** — Kaja Gruntkowska, Yassine Maziane, Zheng Qu, et al., Preprint, 2025. Updates a randomly chosen subset of layers per step rather than the whole network, with convergence proofs and a reported speedup on convolutional models.
- **[AuON: A Linear-time Alternative to Semi-Orthogonal Momentum Updates](https://arxiv.org/abs/2509.24320)** — Dipan Maity, Preprint, 2025. Replaces orthogonalization with a linear-time hyperbolic-cosine RMS scaling inside a spectral-norm trust region, claiming comparable results without any Newton–Schulz iteration.
- **[Conda: Column-Normalized Adam for Training Large Language Models Faster](https://arxiv.org/abs/2509.24218)** — Junjie Wang, Pan Zhou, Yiming Dong, et al., Preprint, 2025. Projects updates into an orthogonal subspace and then applies column-wise second-moment normalization, reporting gains over both AdamW and Muon on LLaMA and GPT-2 pretraining.
- **[AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal Updates](https://arxiv.org/abs/2509.02981)** — Minxin Zhang, Yuxuan Liu, Hayden Schaeffer, Preprint, 2025. Keeps the update exactly orthogonal while scaling it by accumulated gradient norms, adding a single scalar of state; evidence is small-scale.
- **[AdaMuon: Adaptive Muon Optimizer](https://arxiv.org/abs/2507.11005)** — Chongjie Si, Debing Zhang, Wei Shen, Preprint, 2025. Adds element-wise second-moment adaptivity on top of the orthogonalized update plus an RMS rescaling so Adam's learning-rate schedules transfer directly. [Code](https://github.com/Chongjie-Si/AdaMuon)
- **[SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training](https://arxiv.org/abs/2505.24749)** — Yehonathan Refael, Guy Smorodinsky, Tom Tirer, et al., NeurIPS 2025, 2025. Uses exact SVD for moment orthogonalization inside a dynamically adapted low-dimensional subspace, avoiding Newton–Schulz approximation error while keeping memory bounded.
- **[COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs](https://arxiv.org/abs/2502.17410)** — Liming Liu, Zhenghao Xu, Zixuan Zhang, et al., Preprint, 2025. Splits the eigenspace, applying SOAP on the leading subspace and Muon on the remainder, to cut optimizer memory without losing quality.

### Orthogonalization and Numerical Algorithms

- **[Hierarchical Muon: Tiled Newton-Schulz Updates for Efficient Muon Optimization](https://arxiv.org/abs/2606.27216)** — Ziyuan Tang, Tianshi Xu, Yousef Saad, et al., Preprint, 2026. Applies the Newton–Schulz map independently to each tile of a partitioned momentum matrix, trading cross-tile spectral coupling for far lower arithmetic and better GPU kernel shapes; the authors are explicit that for a finite tile count this is a different local map, not a convergent approximation to full-matrix Muon. [Code](https://github.com/tang0389/himuon)
- **[Spectral Scaling Laws of Muon](https://arxiv.org/abs/2606.04058)** — Gagik Magakyan, Pablo Parrilo, Asuman Ozdaglar, Preprint, 2026. Measures how the momentum matrix's singular-value spectrum evolves across model scale, finding quantiles settle after a short burn-in to layer- and size-dependent values obeying clean power laws, and warning that some late layers scale steeply enough to project into a Newton–Schulz failure regime at frontier scale; the consequence is an extrapolation from a 77M–2.8B fit rather than an observed failure.
- **[How Much Orthogonalization Does Muon Need?](https://arxiv.org/abs/2606.00371)** — Hua Huang, Preprint, 2026. Replaces the fixed quintic schedule with a cheaper adaptive cubic one that only aims to land the spectrum in a loose band, then uses that deliberately sloppier map to show training quality does not track polar-decomposition accuracy; the author declines to claim a uniformly better update and reports the largest model tested is marginally worse.
- **[Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training](https://arxiv.org/abs/2603.17970)** — Ben S. Southworth, Stephen Thomas, Preprint, 2026. Swaps the polar-factor iteration for a cheaper triangular, Cholesky-style whitening map for which row-orthonormal matrices are fixed points, accepting a slightly worse per-step direction for much lower optimizer overhead; the advantage is overhead-driven and therefore hardware- and shape-dependent.
- **[TEON: Tensorized Orthonormalization Beyond Layer-Wise Muon for Large Language Model Pre-Training](https://arxiv.org/abs/2601.23261)** — Ruijie Zhang, Yequan Zhao, Ziyue Liu, et al., Preprint, 2026. Breaks the one-layer-at-a-time assumption by stacking several layers into a higher-order tensor and orthogonalizing them jointly, with a convergence bound improving in the number of stacked layers; the bound is an upper bound rather than a measured speedup.
- **[IFNSO: Iteration-Free Newton-Schulz Orthogonalization](https://arxiv.org/abs/2602.02500)** — Chen Hu, Qianxi Zhao, Xiaochen Yuan, et al., Preprint, 2026. Collapses the repeated Newton–Schulz iteration into a single polynomial by scoring each matrix power's contribution, discarding negligible terms, and fitting learnable coefficients for the survivors. (Titled "UNSO: Unified Newton Schulz Orthogonalization" in earlier versions.) [Code](https://github.com/greekinRoma/Unified_Newton_Schulz_Orthogonalization)
- **[Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning](https://arxiv.org/abs/2512.04632)** — Thibaut Boissin, Thomas Massena, Franck Mamalet, et al., Preprint, 2025. Uses almost-orthogonal-Lipschitz preconditioning to supply a better initial guess for the polar factor, removing one Newton–Schulz step.
- **[Beyond the Ideal: Analyzing the Inexact Muon Update](https://arxiv.org/abs/2510.19933)** — Egor Shulgin, Sultan AlRashed, Francesco Orabona, et al., Preprint, 2025. The first analysis of the *approximate* orthogonalization Muon actually performs, showing the Newton–Schulz iteration count must be co-tuned with the learning rate and momentum rather than treated as an implementation detail.
- **[Towards understanding of orthogonalization in Muon](https://openreview.net/forum?id=4vzhqq5hpX)** — Valentyn Boreiko, Zhiqi Bu, Sheng Zha, ICML 2025 TTODLer-FM workshop. Ablates what orthogonalization buys, proposing tile-wise orthogonalization of weight matrices and showing a single learning rate transfers under spectral regularization when depth, width, and tokens are co-scaled; a workshop-length study at nanoGPT scale with tensor-parallel splits up to 16x.
- **[Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials](https://arxiv.org/abs/2506.10935)** — Ekaterina Grishina, Matvey Smirnov, Maxim Rakhuba, Preprint, 2025. Derives Chebyshev-optimal iteration coefficients via the alternance theorem and a Remez procedure, replacing hand-tuned coefficients with a principled family.
- **[The Polar Express: Optimal Matrix Sign Methods and their Application to the Muon Algorithm](https://arxiv.org/abs/2505.16932)** — Noah Amsel, David Persson, Christopher Musco, et al., ICLR 2026 (Oral), 2025. Derives a provably optimal iteration-varying polynomial for the matrix sign and polar factor by solving a minimax problem at each step, giving a faster GPU-friendly drop-in for Newton–Schulz; optimality is for the polynomial subproblem, not end-to-end training. [Code](https://github.com/NoahAmsel/PolarExpress)
- **[Iterative Orthogonalization Scaling Laws](https://arxiv.org/abs/2505.04005)** — Devan Selvaraj, Preprint, 2025. Argues Newton–Schulz orthogonalization degrades as matrices grow because random-matrix singular values shrink, and explicitly declines to propose a remedy — an early statement of the frontier-scale numerical concern.

### Distributed and Communication-Efficient Training

- **[MatrixFSDP: communication-free matrix optimizers under ZeRO-3 parameter sharding](https://arxiv.org/abs/2607.05895)** — Ming Gao, Yanwu Xu, Hao Zhang, Preprint, 2026. Resolves the conflict between whole-matrix updates and ZeRO-3 sharding by deliberately unbalancing the shards so one rank owns each 2D weight in full, making the routine backward reduction deliver the optimizer's input locally and removing the optimizer-step collective entirely; the quality evidence is an exact-match check against a DDP reference rather than an independent comparison.
- **[DMuon: Efficient Distributed Muon Training with Near-Adam Overhead](https://arxiv.org/abs/2606.27153)** — Vincent Chen, Starrick Liu, Regis Cheng, et al., Preprint, 2026. An engineering-focused drop-in distributed module that removes the framework surgery Muon normally requires, bringing per-step overhead to roughly AdamW levels across 8 to 256 GPUs; it reports throughput only, with no convergence or quality comparison. [Code](https://github.com/X-Square-Robot/dmuon)
- **[SignMuon: Communication-Efficient Distributed Muon Optimization](https://arxiv.org/abs/2605.16311)** — Neel Mishra, Kushagara Trivedi, Pawan Kumar, Preprint, 2026. Compresses distributed Muon to one bit per coordinate by having each worker take a local orthogonalized step and ship only the entrywise signs, combined by majority vote, with a convergence proof under spectral-norm smoothness; the systems evidence is four GPUs on small models.
- **[Dion2: A Simple Method to Shrink Matrix in Muon](https://arxiv.org/abs/2512.16928)** — Kwangjun Ahn, Noah Amsel, John Langford, Preprint, 2025. Cuts orthonormalization cost by randomly sampling a fraction of rows or columns each step and running Newton–Schulz only on that sub-block; the authors call the evaluation preliminary and argue the communication savings analytically. [Code](https://github.com/microsoft/dion)
- **[FedMuon: Accelerating Federated Learning with Matrix Orthogonalization](https://arxiv.org/abs/2510.27403)** — Junkang Liu, Fanhua Shang, Junchao Zhou, et al., Preprint, 2025. Adds momentum aggregation and local-global alignment to stop client drift arising from per-client orthogonalization under non-IID data. (One of three distinct papers named FedMuon; disambiguate by author.)
- **[MuonBP: Faster Muon via Block-Periodic Orthogonalization](https://arxiv.org/abs/2510.16981)** — Ahmed Khaled, Kaan Ozkara, Tao Yu, et al., ICLR 2026, 2025. Orthogonalizes per-device matrix shards independently most of the time and performs a full orthogonalization only periodically, using two stepsizes to keep stability; evaluated at 8B with eight-way tensor parallelism, with larger-scale figures given analytically.
- **[On Provable Benefits of Muon in Federated Learning](https://arxiv.org/abs/2510.03866)** — Xinwen Zhang, Hongchang Gao, Preprint, 2025. Shows the orthonormalized direction makes the learning rate independent of problem constants and absorbs heavy-tailed noise without gradient clipping in the federated setting. (One of three distinct papers named FedMuon.)
- **[DeMuon: A Decentralized Muon for Matrix Optimization over Graphs](https://arxiv.org/abs/2510.01377)** — Chuan He, Shuyi Ren, Jingwei Mao, et al., Preprint, 2025. Extends Muon to decentralized graph topologies by combining Newton–Schulz orthogonalization with gradient tracking, matching centralized complexity bounds.
- **[Error Feedback for Muon and Friends](https://arxiv.org/abs/2510.00643)** — Kaja Gruntkowska, Alexander Gaponov, Zhirayr Tovmasyan, et al., ICLR 2026, 2025. Extends error feedback beyond the Euclidean setting, giving what the authors describe as the first distributed linear-minimization-oracle optimizer with convergence guarantees and bidirectional compression.
- **[FedMuon: Federated Learning with Bias-corrected LMO-based Optimization](https://arxiv.org/abs/2509.26337)** — Yuki Takezawa, Anastasia Koloskova, Xiaowen Jiang, et al., Preprint, 2025. Shows naive use of Muon inside FedAvg cannot converge because the linear minimization oracle is nonlinear, then repairs it with control variates. (One of three distinct papers named FedMuon.)
- **[MuLoCo: Muon is a practical inner optimizer for DiLoCo](https://arxiv.org/abs/2505.23725)** — Benjamin Thérien, Xiaolong Huang, Aaron Defazio, et al., Preprint, 2025. Shows that swapping AdamW for Muon as DiLoCo's inner optimizer produces more directionally consistent pseudogradients as worker count grows, addressing DiLoCo's degradation at higher worker counts; the largest figure quoted is an extrapolation. [Code](https://github.com/facebookresearch/MuLoCo)
- **[Dion: Distributed Orthonormalized Updates](https://arxiv.org/abs/2504.05295)** — Kwangjun Ahn, Byron Xu, Natalie Abreu, et al., Preprint, 2025. Replaces Newton–Schulz with amortized power iteration over a low-rank momentum buffer plus error feedback, so orthonormalized updates compose with sharded weights instead of requiring full-matrix reconstruction; quality depends on the rank-fraction hyperparameter. (Retitled and substantially expanded since v1.) [Code](https://github.com/microsoft/dion)

### Quantization and Memory Efficiency

- **[MuonQ: Enhancing Low-Bit Muon Quantization via Directional Fidelity Optimization](https://arxiv.org/abs/2605.11396)** — Yupeng Su, Ruijie Zhang, Ziyue Liu, et al., COLM 2026. Makes Muon's momentum state survive 4-bit quantization by combining pre-quantization normalization, a power-iteration decomposition that protects singular-vector directions, and companding.
- **[Achieving low-bit Muon through subspace preservation and grid quantization](https://iclr.cc/virtual/2026/poster/10008183)** — Huaijin Wu, Bingrui Li, Yebin Yang, et al., ICLR 2026, 2026. Locates Muon's quantization error in the top singular subspace and in outlier patterns, then preserves that subspace while grid-quantizing the remainder to reach 4-bit optimizer state; reports parity with full-precision Muon on LLaMA-architecture pretraining from 130M to 1.1B. [Code](https://github.com/wuhuaijin/lowbit-Muon)
- **[A Convergence Analysis of Adaptive Optimizers under Floating-point Quantization](https://arxiv.org/abs/2510.21314)** — Xuan Tang, Jichu Li, Difan Zou, Preprint, 2025. Finds Muon requires weaker quantization-error control than Adam, whose bound degrades as the second-moment decay approaches one, suggesting greater robustness in low precision.
- **[Unbiased Gradient Low-Rank Projection](https://arxiv.org/abs/2510.17802)** — Rui Pan, Yang Luo, Yuxing Liu, et al., Preprint, 2025. Combines layerwise sampling with Muon to make low-rank gradient projection unbiased, reporting improvements over prior projection methods.
- **[Effective Quantization of Muon Optimizer States](https://arxiv.org/abs/2509.23106)** — Aman Gupta, Rafael Celente, Abhishek Shivanna, et al., Preprint, 2025. Shows Muon's optimizer state can be blockwise-quantized to 8 bits with essentially no loss penalty, and argues Muon is structurally friendlier to quantization than AdamW because a plain linear scheme suffices where AdamW needs dynamic scaling; the gains are in memory footprint only, and later versions substantially revised the experimental scale and headline figure.
- **[Beyond Outliers: A Study of Optimizers Under Quantization](https://arxiv.org/abs/2509.23500)** — Georgios Vlassis, Saleh Ashkboos, Alexandra Volkova, et al., Preprint, 2025. Compares six optimizers from 50M to 1.5B under post-training and quantization-aware training, finding that standard outlier metrics fail to predict post-training-quantization outcomes.
- **[LiMuon: Light and Fast Muon Optimizer for Large Models](https://arxiv.org/abs/2509.14562)** — Feihu Huang, Yuning Luo, Songcan Chen, Preprint, 2025. Replaces the full momentum matrix with a randomized-SVD low-rank factorization and adds variance reduction, proving a sample complexity bound under both standard and generalized smoothness; the authors state the empirical results do not reach state of the art.
- **[Low-rank Orthogonalization for Large-scale Matrix Optimization](https://arxiv.org/abs/2509.11983)** — Chuan He, Zhanwang Deng, Zhaosong Lu, Preprint, 2025. Orthogonalizes only the low-rank part of the gradient, with complexity bounds covering heavy-tailed noise, evaluated on GPT-2 and LLaMA pretraining.
- **[Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](https://arxiv.org/abs/2506.19697)** — Jungwoo Park, Taewhoo Lee, Chanwoong Yoon, et al., Preprint, 2025. Uses Muon together with a single-scale normalization to prevent activation outliers forming during pretraining at all, arguing outliers are a training-strategy artefact rather than intrinsic to transformers.

### Fine-Tuning, Post-Training, and Optimizer Transfer

- **[When Does Muon Help Agentic Reinforcement Learning?](https://arxiv.org/abs/2607.16169)** — Kai Ruan, Jinghao Lin, Zihe Huang, et al., Preprint, 2026. Probes whether the pretraining advantage carries into sparse-reward agent training by applying Muon to a small instruction-tuned policy's hidden matrices, concluding the outcome depends jointly on the advantage estimator and the learning rate; results are single-seed at 0.5B on one benchmark.
- **[LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold](https://arxiv.org/abs/2606.12921)** — Franz Louis Cesista, Katherine Crowson, Cédric Simal, et al., Preprint, 2026. Rederives the spectral steepest-descent rule for the geometry of LoRA's factored updates, producing an optimizer whose tuned learning rate carries across rank, width, and depth while avoiding QR factorization and second-moment state.
- **[Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR](https://arxiv.org/abs/2605.19282)** — Chongyu Fan, Gaowen Liu, Mingyi Hong, et al., Preprint, 2026. Identifies post-training regimes — vision-language-action training and reinforcement learning with verifiable rewards — where flattening every singular value to 1 actively hurts, and replaces it with a two-stage filter that keeps dominant directions and suppresses noisy ones.
- **[Can Muon Fine-tune Adam-Pretrained Models?](https://arxiv.org/abs/2605.10468)** — Xingyu Qu, Peigeng Huang, Samuel Horváth, ICML 2026, 2026. Diagnoses why switching an Adam-pretrained checkpoint to Muon hurts — the two carry different implicit biases, and the resulting update disturbs pretrained knowledge in proportion to its magnitude — then shows shrinking the update via LoRA largely closes the gap; the symmetric Muon-pretrained control is only 561M.
- **[LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters](https://arxiv.org/abs/2507.12142)** — Vladimir Bogachev, Vladimir Aletov, Alexander Molozhavenko, et al., ICLR 2026, 2025. Treats LoRA adapters as points on the fixed-rank matrix manifold and optimizes them there directly, so training no longer depends on the arbitrary choice of low-rank factorization; evaluated on Llama 3 8B and 1B commonsense and math benchmarks plus subject-driven image generation. (Titled "RiemannLoRA: A Unified Riemannian Framework for Ambiguity-Free LoRA Optimization" in v1.)
- **[POME: Post Optimization Model Edit via Muon-style Projection](https://arxiv.org/abs/2510.06627)** — Yong Liu, Di Fu, Yang Luo, et al., Preprint, 2025. Applies a Muon-style truncated-SVD projection to the fine-tuned-minus-pretrained weight delta *after* training, needing no extra data — a use of the operator outside the optimizer loop entirely.
- **[REG: A Regularization Optimizer for Robust Training Dynamics](https://arxiv.org/abs/2510.03691)** — Zehua Liu, Han Wu, Xiaojin Fu, et al., Preprint, 2025. Argues the matrix-sign operator is too aggressive and substitutes a gentler row-and-column scaling grounded in matrix-equilibration theory, targeting compatibility with AdamW-pretrained checkpoints; the authors state a convergence proof for the full algorithm is open and that their best-performing norm choice contradicts classical expectations.
- **[Leveraging Coordinate Momentum in SignSGD and Muon: Memory-Optimized Zero-Order LLM Fine-Tuning](https://arxiv.org/abs/2506.04430)** — Egor Petrov, Grigoriy Evseev, Aleksey Antonov, et al., Preprint, 2025. Brings matrix structure into zeroth-order fine-tuning at constant function-evaluation cost, reporting substantial memory reduction relative to full fine-tuning.

### Empirical Benchmarks and Applications

- **[Muon Meets Mamba: Spectral Optimization for State Space Models](https://arxiv.org/abs/2608.03941)** — Arslan Battalov, Karim Kramin, Alexander Markotenko, et al., Preprint, 2026. Asks which weight matrices inside a state-space model actually benefit from orthogonalized updates, finding output-projection-only application beats input-projection-only, and reporting that improved conditioning does *not* explain the measured difference.
- **[CMuon: Accelerating and Stabilizing Diffusion Transformer Training via Chunked Momentum Orthogonalization](https://arxiv.org/abs/2608.02502)** — Chuyan Chen, Peng Sun, Kun Yuan, ECCV 2026. Identifies that diffusion-transformer weight tensors fuse semantically distinct projections into one matrix, which undermines whole-matrix orthogonalization, and repairs it by partitioning into sub-blocks orthogonalized independently.
- **[Beyond Adam: SOAP and Muon for Faster, Label-Efficient Training of Machine Learning Interatomic Potentials](https://arxiv.org/abs/2607.02499)** — Gil Harari, Yoel Zimmermann, Ola Tangen Kulseng, et al., Preprint, 2026. Benchmarks matrix-structured optimizers against the field's Adam default for neural interatomic potentials, finding Muon's benefit narrower and less consistent than SOAP's.
- **[MuonSSM: Orthogonalizing State Space Models for Sequence Modeling](https://arxiv.org/abs/2606.30461)** — Thai-Khanh Nguyen, Ngoc-Bich-Uyen Vo, Thieu N. Vo, et al., ICML 2026 (Oral). Moves Newton–Schulz orthogonalization out of the optimizer and into the architecture, applying it to low-rank input injections so memory updates stay spectrally conditioned without breaking parallel-scan complexity.
- **[Muon in Vision Transformers: Optimizer-Recipe Interactions and Gradient Spectra](https://arxiv.org/abs/2605.24770)** — Ben S. Southworth, Shuai Jiang, Daniel McBride, et al., Preprint, 2026. Shows the advantage over AdamW is not recipe-neutral — it grows disproportionately with aggressive data augmentation — and traces the mechanism to how singular-value energy is distributed across QKV and deep MLP-down gradient matrices.
- **[Benchmarking Optimizers for MLPs in Tabular Deep Learning](https://arxiv.org/abs/2604.15297)** — Yury Gorishniy, Ivan Rubachev, Dmitrii Feoktistov, et al., Preprint, 2026. Runs fifteen optimizers across seventeen tabular datasets under one shared protocol and lands on Muon as the reliable choice over AdamW for MLP backbones, conditioned explicitly on the training-efficiency overhead being affordable. [Code](https://github.com/yandex-research/tabular-dl-optimizers)
- **[MuonRec: Shifting the Optimizer Paradigm Beyond Adam in Scalable Generative Recommendation](https://arxiv.org/abs/2603.00416)** — Rong Shan, Aofan Yu, Bo Chen, et al., Preprint, 2026. Ports the orthogonalized-momentum update into recommender training, reporting fewer converged steps and better ranking quality; the datasets are small relative to the 0.5B–3B backbones, and no wall-clock accounting is given.
- **[Muon with Spectral Guidance: Efficient Optimization for Scientific Machine Learning](https://arxiv.org/abs/2602.16167)** — Binghang Lu, Jiahao Zhang, Guang Lin, Preprint, 2026. Splits matrix gradients into singular modes and attaches a per-mode relaxed scalar auxiliary variable rule, evaluated on physics-informed neural networks and PDE benchmarks at small scale.
- **[What Really Matters in Matrix-Whitening Optimizers?](https://arxiv.org/abs/2510.25000)** — Kevin Frans, Pieter Abbeel, Sergey Levine, Preprint, 2025. Decomposes matrix-whitening optimizers into spectral normalization and variance adaptation and shows under symmetric per-optimizer tuning that the variance-adaptation half, which Muon omits, explains more of the gain over Adam than spectral accuracy does; the study is one architecture at one scale and the absolute effect is small. [Code](https://github.com/kvfrans/matrix-whitening)
- **[Optimization Benchmark for Diffusion Models on Dynamical Systems](https://arxiv.org/abs/2510.19376)** — Fabian Schaipp, Preprint, 2025. Extends optimizer benchmarking outside language modelling by training a diffusion model to denoise Navier–Stokes trajectories, with learning rate and weight decay grid-searched separately per optimizer; the reported advantage is epoch-budget-matched, and Muon costs about 1.45× AdamW per step. [Code](https://github.com/fabian-sp/sda)
- **[The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](https://arxiv.org/abs/2510.09378)** — Natalie Abreu, Sham Kakade, Nikhil Vyas, et al., Preprint, 2025. Measures how much iteration-complexity headroom cheap approximations such as Muon and SOAP leave on the table by running full Gauss–Newton preconditioning at small scale, and finds most of the benefit is recoverable from layerwise Hessian information; the gains are in iterations at 45M–150M, and full Gauss–Newton is not deployable.
- **[Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046)** — Kaiyue Wen, David Hall, Tengyu Ma, et al., ICLR 2026, 2025. Re-runs eleven pretraining optimizers under equal-density hyperparameter tuning and multiple evaluation points, finding reported speedups over AdamW are systematically inflated by under-tuned baselines and short-horizon evaluation, with the matrix-method advantage shrinking from roughly 1.4× at 0.1B to 1.1× at 1.2B. [Code](https://github.com/marin-community/marin/tree/kaiyue/optimizers)
- **[Benchmarking Optimizers for Large Language Model Pretraining](https://arxiv.org/abs/2509.01440)** — Andrei Semenov, Matteo Pagliardini, Martin Jaggi, Preprint, 2025. A uniformly tuned comparison across model size, batch size, and training duration spanning roughly 2,900 trained models, finding vanilla Muon weak at small batch sizes but the weight-decayed variant consistently robust, and showing that the learning-rate decay floor alone reshuffles optimizer rankings. [Code](https://github.com/epfml/llm-optimizer-benchmark)
- **[Muon Optimizer Accelerates Grokking](https://arxiv.org/abs/2504.16041)** — Amund Tveit, Bjørn Remseth, Arve Skogvold, Preprint, 2025. Reports that training with Muon rather than AdamW makes delayed generalization set in substantially earlier across seven small algorithmic tasks; the setting is tiny and the mechanism is not isolated. [Code](https://github.com/atveit/torch_grokking)

### Generalization, Robustness, and Regularization

- **[Sharpness-Aware Minimization and Muon: Robustness under the Spectral Norm](https://arxiv.org/abs/2607.26001)** — Wenzhi Zhong, Edward Milsom, Michael Murray, Preprint, 2026. Makes both stages of sharpness-aware minimization matrix-aware by measuring the worst-case perturbation in a layerwise spectral geometry and pairing it with a Muon outer step, then identifies which geometry pairings actually pay off.
- **[Scale Weight Decay and Train Better](https://arxiv.org/abs/2607.23777)** — Anuj Apte, Preprint, 2026. Proposes scaling weight decay by the current learning-rate fraction rather than holding it constant, which removes a stationarity bias for both SGD and Muon, evaluated on mixture-of-experts models from 72M to 930M.
- **[Muon Learns More Robust and Transferable Features than Adam](https://arxiv.org/abs/2606.09658)** — Tianyu Ruan, Fengzhuo Zhang, Shuche Wang, et al., Preprint, 2026. Shifts the question from training speed to representation quality, comparing pretrained models on corruption robustness, hidden-state effective rank, and downstream transfer, and backing the pattern with a proof in a stylized feature-learning model.
- **[MiMuon: Mixed Muon Optimizer with Improved Generalization for Large Models](https://arxiv.org/abs/2605.19619)** — Feihu Huang, Yuning Luo, Songcan Chen, Preprint, 2026. Derives a generalization bound for Muon, shows it is loose in the iteration count, and improves it by mixing Muon with momentum SGD without slowing convergence.
- **[When Muon Optimizer Meets Adversarial Training: A Theoretical and Empirical Study](https://arxiv.org/abs/2605.26929)** — Jun Yan, Weiquan Huang, Jiankai Zuo, et al., Preprint, 2026. Tests whether substituting Muon changes the robustness outcome of adversarial training across several architectures and threat models; gains are architecture-dependent, competitive with rather than better than SGD on convolutional networks, with clear wins mainly over AdamW.
- **[How Muon's Spectral Design Benefits Generalization: A Study on Imbalanced Data](https://arxiv.org/abs/2510.22980)** — Bhavya Vasudeva, Puneesh Deora, Yize Zhao, et al., Preprint, 2025. Explains the generalization edge through an idealized spectral gradient descent that learns all principal components at equal rates rather than prioritizing dominant ones, an effect amplified by depth and most visible on imbalanced data; the core theory assumes joint diagonalizability and population statistics.
- **[Noise-Adaptive Layerwise Learning Rates: Accelerating Geometry-Aware Optimization for Deep Neural Network Training](https://arxiv.org/abs/2510.14009)** — Jie Hao, Xiaochuan Gong, Jie Xu, et al., Preprint, 2025. Adds a per-layer learning rate on top of geometry-aware optimizers such as Muon by estimating gradient variance on the fly in the dual norm induced by the chosen oracle, replacing fixed rates shared across a norm group.
- **[Cautious Weight Decay](https://arxiv.org/abs/2510.12402)** — Lizhang Chen, Jonathan Li, Kaizhao Liang, et al., ICLR 2026, 2025. A one-line, optimizer-agnostic change applying decoupled weight decay only where update and parameter signs agree, preserving the original loss's stationary manifold; Muon is one of several host optimizers rather than the focus.
- **[Training Transformers with Enforced Lipschitz Constants](https://arxiv.org/abs/2507.13338)** — Laker Newhouse, R. Preston Hess, Franz Cesista, et al., Preprint, 2025. Maintains norm-constrained weights throughout training and finds Muon strictly improves the performance-versus-Lipschitz frontier relative to AdamW at scales from 2M to 145M parameters.

### Limitations, Counterexamples, and Negative Results

- **[Post-Grokking Collapse at the Representation-Readout Interface in Muon-Trained Transformers](https://arxiv.org/abs/2608.07436)** — Ali Janati, Kaoutar El Maghraoui, Andrei Kanavalau, et al., Preprint, 2026. Documents that Muon-trained transformers which have already generalized subsequently lose that generalization under continued training in all nine configurations tested, localizes the fault to the embedding/readout interface, and shows freezing that interface removes it; the setting is one synthetic task at small scale.
- **[On MUON optimization: From non-convergence to an error analysis with Polar Express and the Newton-Schulz polynomial from implementations](https://arxiv.org/abs/2608.04607)** — Thang Do, Steffen Dereich, Arnulf Jentzen, Preprint, 2026. Generalizes Muon to arbitrary-degree Newton–Schulz-type polynomials, proves it can fail to converge on a simple class of stochastic problems for almost every mini-batch size, and supplies matching error rates in step count and batch size.
- **[Reassessing Muon for Matrix Factorization](https://arxiv.org/abs/2607.13246)** — Ali Parviz, Gal Mishne, Alex Cloninger, Preprint, 2026. Strips away scale, architecture, and data confounds by testing on low-rank matrix factorization and reports that Muon does not consistently outperform a tuned AdamW there, with several previously reported advantages sensitive to hyperparameter choices.
- **[The Active Ingredient in Muon's Grokking](https://arxiv.org/abs/2607.20512)** — Yufeng Wang, Preprint, 2026. Ablates Muon into components on modular-arithmetic grokking and isolates orthogonalization rather than spectral scaling as the source of the speedup — spectral scaling alone is no faster than AdamW — while showing that reducing Newton–Schulz iterations makes the solution fragile and that under a stability-aware metric the "faster" claim can invert.
- **[Towards Understanding the Power and Limits of the Muon Optimizer: A River-Valley Perspective](https://arxiv.org/abs/2606.21514)** — Tianqi Shen, Jinji Yang, Runze Shi, et al., Preprint, 2026. Builds a trajectory-level account of why the early-training advantage does not carry through to late training, proving that near the valley floor Muon progresses more slowly than gradient descent and is prone to overshoot and oscillation because the orthogonalized update discards residual scale, and recommends switching to a gradient-descent-like refiner near convergence.
- **[Muon is Not That Special: Random or Inverted Spectra Work Just as Well](https://arxiv.org/abs/2605.11181)** — Zakhar Shumaylov, Nathaël Da Costa, Peter Zaika, et al., Preprint, 2026. Attacks the geometry explanation directly by exhibiting an optimizer that discards the singular values and substitutes random noise yet matches Muon, arguing the real drivers are gradient alignment and step-size optimality; the parity result is empirical at 124M on a small corpus.
- **[Muon Does Not Converge on Convex Lipschitz Functions](https://arxiv.org/abs/2605.08980)** — Tetiana Parshakova, Ahmed Khaled, Michael Crawshaw, et al., Preprint, 2026. Constructs explicit counterexamples proving Muon fails to converge on convex Lipschitz functions under any learning-rate schedule, shows error feedback provably repairs convergence, and then demonstrates that this theoretically correct fix makes training worse in practice — concluding the convex Lipschitz model is the wrong lens.
- **[To Use or not to Use Muon: How Simplicity Bias in Optimizers Matters](https://arxiv.org/abs/2603.00742)** — Sara Dragutinović, Yedi Zhang, Rajesh Ranganath, Preprint, 2026. Argues Muon buys speed by flattening the sequential, low-rank-first learning order gradient descent exhibits, so it forfeits simplicity bias and can latch onto spurious features and fail to share structure across tasks; the theorems are gradient-flow results for two-layer linear networks, with small-scale empirical support.
- **[A Minimalist Optimizer Design for LLM Pretraining](https://arxiv.org/abs/2506.16659)** — Athanasios Glentis, Jiaxiang Li, Andi Han, et al., Preprint, 2025. Asks how little optimizer state suffices and reports that column-normalized SGD with momentum only on the last layer matches Muon and Adam at a fraction of Adam's memory — a direct challenge to how much optimizer machinery the gains require.

### Adjacent Matrix and Spectral Optimizers

Work where Muon is context, a baseline, or one member of a family rather than the
subject.

- **[The Loss Does Not See the Basis, but Adam Does](https://arxiv.org/abs/2608.05136)** — Devender Singh, Preprint, 2026. Traces gradient flow's low-rank implicit bias to gauge symmetry of the factored loss and shows only gauge-equivariant optimizers — Muon and Shampoo among them — can inherit it, while noting equivariance is necessary but not sufficient and that Muon's edge fades as the spectral tail grows. [Code](https://github.com/idevender/loss-basis-adam)
- **[Clarifying Shampoo: Adapting Spectral Descent to Stochasticity and the Parameter Trajectory](https://arxiv.org/abs/2602.09314)** — Runa Eschenhagen, Anna Cai, Tsung-Hsien Lee, et al., Preprint, 2026. Explains Shampoo as time-averaged semi-orthogonal spectral descent, clarifying how it differs from Muon once stochasticity is accounted for.
- **[PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective](https://arxiv.org/abs/2505.21799)** — Tim Tsz-Kit Lau, Qi Long, Weijie Su, Preprint, 2025. Separates curvature anisotropy, which Adam addresses, from gradient anisotropy, which Muon addresses, and derives a polar-decomposition preconditioner scaled by the gradient's nuclear norm to restore the magnitude sensitivity a sign-like step discards. [Code](https://github.com/timlautk/polargrad)
- **[GradPower: Powering Gradients for Faster Language Model Pre-Training](https://arxiv.org/abs/2505.24275)** — Jinbo Wang, Mingze Wang, Jiaqi Zhang, et al., Preprint, 2025. An element-wise sign-power gradient transform that composes with Muon as readily as with Adam, evaluated from 66M to 2B including mixture-of-experts models.
- **[ASGO: Adaptive Structured Gradient Optimization](https://arxiv.org/abs/2503.20762)** — Kang An, Yuxing Liu, Rui Pan, et al., Preprint, 2025. Builds a preconditioner from structured matrix gradients with convergence rates exploiting low-rank gradients and block-diagonal Hessians. [Code](https://github.com/infinity-stars/ASGO)
- **[MARS: Unleashing the Power of Variance Reduction for Training Large Models](https://arxiv.org/abs/2411.10438)** — Huizhuo Yuan, Yifeng Liu, Shuang Wu, et al., ICML 2025, 2024. Combines preconditioned gradient methods with a scaled stochastic recursive momentum estimator so variance reduction helps at language-model scale, instantiated for AdamW, Lion, and Shampoo. [Code](https://github.com/AGI-Arena/MARS)

## Implementations and Ecosystem

Only first-party implementations and official documentation are listed. Details
below were checked on 2026-08-13/14 and change frequently — verify against the
source before relying on them.

- **[KellerJordan/Muon](https://github.com/KellerJordan/Muon)** — the reference implementation. Provides `Muon`, `SingleDeviceMuon`, `MuonWithAuxAdam`, and `SingleDeviceMuonWithAuxAdam`. Applies to parameters with `ndim >= 2`; the README directs embeddings, classifier heads, and hidden gains and biases to AdamW. Distributed support is DDP-style `all_gather` with round-robin work assignment across ranks — there is no FSDP, ZeRO, or tensor-parallel support. Newton–Schulz runs in `bfloat16` with quintic coefficients `(3.4445, -4.7750, 2.0315)` over 5 steps, the update is scaled by `max(1, A/B)^0.5`, and decoupled weight decay is applied before the update. Companion speedrun repository: [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
- **[torch.optim.Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)** — official PyTorch implementation, first shipped in PyTorch 2.9 ([source](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py)). Applies to 2D parameters only; the documentation directs biases and embeddings to a standard method such as AdamW, constructed as a separate optimizer instance. **Single-device only** — there is no DTensor or distributed logic in the module. `adjust_lr_fn` selects the update-scale convention: `"original"` for `sqrt(max(1, A/B))` or `"match_rms_adamw"` for `0.2 * sqrt(max(A, B))`.
- **[DeepSpeed](https://www.deepspeed.ai/docs/config-json/)** — Muon is a natively supported optimizer type, enabled by `"optimizer": {"type": "Muon", ...}`. A parser tags 2D hidden matrices for Muon during engine init and routes embeddings, normalization parameters, biases, and the output head to Adam inside the same hybrid optimizer, with separate `muon_lr` and `adam_lr`. **Supports ZeRO stages 1, 2, and 3**; the Muon update is moved into the stage-1/2 flat-partition path so per-parameter gradients stay unflattened. `ns_method` selects between a Gram-matrix and a standard Newton–Schulz variant. See also the [PyTorch blog write-up](https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/).
- **[NVIDIA Megatron-Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.muon.html)** — first-class Muon in the optimizer config plus a layer-wise distributed optimizer that chains Muon and AdamW. Applies to 2D weights, with `muon_scalar_optimizer` (Adam or Lion) handling embeddings, biases, and norms, and `muon_split_qkv=True` splitting fused QKV weights before orthogonalization. Data-parallel sharding assigns whole layers per rank so the preconditioner is computed locally; tensor parallelism is handled by `muon_tp_mode` ∈ `{duplicated, distributed, blockwise}`, which trade communication against exactness — `blockwise` avoids collectives but orthogonalizes only per block. The algorithm now lives in `emerging_optimizers`, with `core.optimizer.muon` a backward-compatible shim.
- **[NVIDIA-NeMo/Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)** — the first-party library holding the Muon kernel used by Megatron-Core, alongside Shampoo and SOAP. Asserts all parameters are 2D and directs embeddings, the final layer, and 1D parameters to AdamW. Offers selectable Newton–Schulz coefficient families (`simple`, `quintic`, `polar_express`, `cans`, `aol`, `deepseekv4`, `cubic5`, `custom`) and three scale modes (`shape_scaling`, `spectral`, `unit_rms_norm`). APIs are marked experimental.
- **[NVIDIA NeMo-RL Muon guide](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html)** — exposes Megatron-Core's Muon for supervised fine-tuning and RL post-training, marked experimental. Requires the Megatron backend; FSDP2 is not supported. NVIDIA's own documentation reports only minor gains over Adam when post-training Adam-pretrained models.
- **[MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight)** — Moonshot AI's scaled Muon recipe accompanying [arXiv:2502.16982](https://arxiv.org/abs/2502.16982): weight decay, per-parameter update-scale adjustment via `0.2 * sqrt(max(A, B))`, and a ZeRO-1-style distributed implementation. Applies to `ndim >= 2` excluding `embed_tokens` and `lm_head`. Ships the Moonlight-16B-A3B checkpoints. Note that `github.com/MoonshotAI/Muon` does not exist.
- **[microsoft/dion](https://github.com/microsoft/dion)** — efficient implementations of orthonormal optimizers for distributed training, shipping Dion2, Dion, Muon, and NorMuon. Built on PyTorch DTensor with support for single device, DDP, FSDP2, and hybrid sharding; tensor parallelism is supported for Dion but not for the bundled Muon. Orthonormal updates apply to 2D matrices, with Lion or AdamW for scalars.
- **[optax.contrib.muon](https://optax.readthedocs.io/en/latest/_collections/examples/contrib/muon.html)** — a JAX implementation in the official Optax repository, under the experimental `contrib` module rather than the core API. 2D parameters get Muon by default and all non-2D parameters are routed automatically to a built-in AdamW; rank > 2 tensors require explicit dimension numbers. No optimizer-level sharding logic.
- **[keras.optimizers.Muon](https://keras.io/api/optimizers/muon/)** — a first-class Keras 3 optimizer with a built-in AdamW fallback inside one optimizer object. Applies to 2D non-excluded variables; all 0D/1D variables, embeddings (`exclude_embeddings=True` by default), the final output dense layer, and any layer matched by `exclude_layers` go to AdamW. Exposes `muon_a/b/c`, `ns_steps`, and an `rms_rate` matching Moonshot's RMS convention. Distributed behaviour is not documented.

**Verified absences as of 2026-08-14.** Hugging Face `transformers` does not list Muon among its Trainer optimizers. `torchtitan` recognizes only Adam and AdamW; its Muon feature request was closed without a merged implementation, citing `torch.optim.Muon` being single-node. Levanter's configuration documentation shows no Muon optimizer.

## Blog Posts and Explanatory Resources

First-party technical writing. These are not peer-reviewed papers and are listed
separately for that reason; several are nonetheless the earliest or clearest
statement of an idea that later appears in the literature. Keller Jordan's original
post is filed under [Origins](#origins-and-foundational-perspectives) because it is
the canonical primary source for the algorithm itself.

- **[Deriving Muon](https://jeremybernste.in/writing/deriving-muon)** — Jeremy Bernstein, 2025. Derives Muon in four steps: metrize the linear layer with RMS norms, bound how weight perturbations move outputs, dualize the gradient under the induced operator norm, then orthogonalize quickly with Newton–Schulz.
- **[Modular Manifolds](https://thinkingmachines.ai/blog/modular-manifolds/)** — Jeremy Bernstein, Thinking Machines Lab, 2025. Introduces manifold Muon, constraining weights to the Stiefel manifold while optimizing under a spectral-norm constraint, and generalizes to composable per-layer manifolds with learning rates budgeted by Lipschitz sensitivity. [Code](https://github.com/thinking-machines-lab/manifolds)
- **[Understanding Muon Chapter 1: Into the Matrix](https://www.lakernewhouse.com/writing/muon-1)** — Laker Newhouse. An introductory walk through the matrix view of the update, aimed at readers meeting Muon for the first time.
- **[Muon, muP, and the Compute-Time Tradeoff](https://www.essential.ai/blog/optimizer)** — Essential AI, 2025. First-party companion to [arXiv:2505.02222](https://arxiv.org/abs/2505.02222), carrying the maximal-update-parameterization framing.
- **[Squeezing 1-2% Efficiency Gains Out of Muon by Optimizing the Newton-Schulz Coefficients](https://leloykun.github.io/ponder/muon-opt-coeffs/)** — Franz Louis Cesista, 2025. The origin of the coefficient-tuning line later formalized by Polar Express and Chebyshev-type schemes.
- **[Muon and a Selective Survey on Steepest Descent in Riemannian and Non-Riemannian Manifolds](https://leloykun.github.io/ponder/steepest-descent-non-riemannian/)** — Franz Louis Cesista, 2025. The conceptual bridge from normed-space steepest descent to manifold formulations.
- **[Deep Learning Optimizers as Steepest Descent in Normed Spaces](https://leloykun.github.io/ponder/steepest-descent-opt/)** — Franz Louis Cesista. An accessible statement of the normed-space view.
- **[Gram Newton-Schulz](https://tridao.me/blog/2026/gram-newton-schulz/)** — Tri Dao, 2026. Iterates on the smaller symmetric Gram matrix rather than the rectangular input, with a hardware-aware analysis across GPU architectures.
- **[Modula documentation](https://docs.modula.systems/)** — Jeremy Bernstein and Laker Newhouse. Living documentation of modular norms and duality maps.
- **[Depths of First-Order Optimization](https://docs.google.com/presentation/d/1PIAChMGGwhmdUxDPyOo1o8Qlhq3h_ofV2mhBb6JHH04)** — Jeremy Bernstein. Slide-form overview of the norm-based view of optimizer design.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

The short version of the standard: every entry needs a **primary source** (an
official venue page, a canonical arXiv abstract page, or an official repository);
venue status is claimed only when an official page confirms it; chronology uses the
**first public version** date, never the latest revision; summaries are written in
your own words and describe scope and limitations rather than selling the result;
each paper lives in exactly one topical section; and no promotional language,
citation-count rankings, or copied abstracts.

## Acknowledgements and License

Thanks to everyone who has contributed entries, corrections, and pull requests to
this list, and to the authors whose work it indexes.

This repository does not currently contain a `LICENSE` file. Until the maintainer
adds one, please treat the curated text here as all-rights-reserved and link to the
original sources rather than redistributing entries. Linked papers, code, and blog
posts remain under their own licenses.
