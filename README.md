# Awesome Muon Optimizer 

<!-- [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) -->
![Papers](https://img.shields.io/badge/papers-60+-blue)
![Last Updated](https://img.shields.io/badge/updated-Nov%202025-green)

> **A curated list of research on Muon** — a spectrum-aware optimizer that orthogonalizes gradient updates for neural network training. This collection documents the theoretical foundations, empirical evaluations, and practical developments around Muon and related spectral/matrix-based optimization methods.

---

##  Table of Contents

- [ Background: Spectral Bias and Adaptive Optimizers](#-background-spectral-bias-and-adaptive-optimizers)
- [ Original Literature](#-original-literature)
- [ Theoretical Analysis](#-theoretical-analysis)
- [ Understanding Property](#-understanding-property)
- [ Critical Batch Size](#️-critical-batch-size)
- [ Empirical Evaluation](#-empirical-evaluation)
- [ Efficient Algorithm](#-efficient-algorithm)
- [ Distributed Setting](#-distributed-setting)
- [ Scaling](#-scaling)
- [ Regularization](#️-regularization)
- [ Enhancement](#-enhancement)
- [ Blog Posts](#-blog-posts)
- [ Related Work and Perspectives](#-related-work-and-perspectives)

---

## 🎓 Background: Spectral Bias and Adaptive Optimizers

> Understanding the motivation for spectrum-aware optimizers like Muon.

<details>
<summary><b>Spectral Bias in Practice: the Role of Function Frequency in Generalization</b></summary>

[Paper Link](https://papers.neurips.cc/paper_files/paper/2022/file/306264db5698839230be3642aafc849c-Paper-Conference.pdf)

- **The phenomenon**: Neural networks exhibit frequency-dependent learning rates during gradient descent. For a function $f$ decomposed into Fourier components $f = \sum_k a_k \phi_k$ (where $\phi_k$ are basis functions at frequency $k$), low-frequency components (small $k$) are learned exponentially faster than high-frequency components (large $k$).
- **Theoretical characterization**: Basri et al. (2019) showed that for a component at frequency $\omega$, the learning rate effectively scales as $\mathcal{O}(1/\omega^2)$. This means a 10x higher frequency component learns ~100x slower.
- **Training dynamics**: During optimization, coarse structures (low-frequency) are captured first, while fine-grained details (high-frequency) emerge only later in training or may not be learned at all with limited training time.
- **Benefits**: On balanced data, this spectral bias acts as implicit regularization—promoting smooth functions that resist overfitting to noise, improving generalization.
- **Limitations**: For imbalanced datasets, minority classes or rare features often correspond to high-frequency signals in the data distribution. These are systematically learned slower and may be neglected, leading to poor minority-class performance.
- **Motivation for Muon**: This limitation directly motivates spectrum-aware optimizers like Muon that balance learning across all frequency components rather than privileging low frequencies.

</details>

<details>
<summary><b>Adam vs. SGD: Achieving Comparable Generalization in Image Classification Through Adaptive Techniques</b></summary>

[Paper Link](https://rjpn.org/ijcspub/viewpaperforall.php?paper=IJCSP25B1033)

- **The generalization gap**: In computer vision tasks (especially ImageNet classification with ResNets), Adam historically achieved 1-3% lower test accuracy than SGD+momentum at similar training loss, despite converging faster.
- **Root cause**: The gap was not fundamental to adaptive methods but due to: (1) lack of decoupled weight decay, and (2) inappropriate scaling across layers.
- **Solution - AdamW**: Decoupling weight decay from gradient-based updates: instead of adding $\lambda w$ to gradient, directly update $w \leftarrow w(1 - \eta\lambda)$ after the adaptive step. This ensures weight decay acts as true L2 regularization regardless of gradient magnitude.
- **Solution - Layer-wise normalization**: Techniques like layer-wise adaptive rate scaling (LARS) or normalization ensure each layer receives appropriate update magnitudes.
- **Result**: With these modifications, AdamW achieves comparable (often identical within error bars) test accuracy to SGD on ImageNet and CIFAR benchmarks.
- **Implication**: The poor generalization of Adam was an artifact of implementation details, not a fundamental limitation of adaptive optimization. This opened the door for spectrum-aware adaptive methods like Muon.

</details>

<details>
<summary><b>Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models</b></summary>

[Paper Link](https://arxiv.org/abs/2210.14199)

- **Core finding**: Models with identical pre-training validation loss can have vastly different downstream task performance. The paper demonstrates this by varying training duration, model size, and optimization method while holding loss constant.
- **Concrete measurement**: An adversarially-trained model (perturbed optimizer) achieved ~6% lower downstream accuracy than a standard AdamW model, despite both having the same pre-training loss. The adversarial model's Hessian trace was ~2x larger.
- **Flatness-performance correlation**: Plotting downstream accuracy vs. Hessian trace (measure of sharpness) across multiple models revealed clear inverse correlation: flatter minima (lower trace) → better downstream performance. The Hessian trace $\text{tr}(H) = \sum_i \lambda_i$ sums eigenvalues of the loss Hessian.
- **Training beyond convergence**: Continuing to train past the point of loss convergence still improved downstream accuracy—implying the solution was becoming flatter/more transferable even though loss didn't change.
- **Mechanism**: Optimizer choice creates implicit bias toward different regions of the loss landscape. Methods like AdamW (with dropout, weight decay) find flatter basins that generalize better to new tasks. Sharper minima memorize training distribution specifics.
- **Relevance to Muon**: Suggests spectrum-aware optimizers like Muon, which have strong implicit regularization (spectral norm constraints), may find flatter solutions that transfer better—a testable hypothesis for future work.

</details>

<details>
<summary><b>Deconstructing What Makes a Good Optimizer for Language Models</b></summary>

[Paper Link](https://arxiv.org/abs/2407.07972)

- **Experimental scope**: Systematic comparison of AdamW, AdaFactor, Lion, and SignSGD on transformer language models up to 1.3B parameters across multiple training scales.
- **Main result - Similar final performance**: When hyperparameters are well-tuned for each optimizer, they reach essentially the same final perplexity and zero-shot downstream accuracy. Differences in final metrics are typically within noise/variance.
- **Plain SGD failure**: Standard SGD (with momentum) fails to train transformers effectively without heavy architectural modifications or extremely careful tuning. Loss either diverges or converges to much worse solutions. This is why adaptive optimizers dominate in NLP.
- **Key differences lie elsewhere**: While final loss is similar, optimizers differ in:
  1. **Hyperparameter robustness**: How sensitive to learning rate, beta values, warmup schedule
  2. **Training efficiency**: How many steps/FLOPs to reach target loss
  3. **Solution characteristics**: Different implicit biases affecting downstream task performance despite same perplexity
- **Implication**: For language models, optimizer choice affects the *nature* of the learned solution and *path* to convergence more than the raw pre-training loss value. This motivates investigating whether spectrum-aware methods like Muon find different (potentially better) solutions for downstream tasks.

</details>

<details>
<summary><b>Scalable Second Order Optimization for Deep Learning (Shampoo)</b> — <i>Rohan Anil et al. (Google)</i></summary>

[Paper Link](https://arxiv.org/abs/2002.09018)

- **The algorithm**: Shampoo preconditions gradients by maintaining two per-layer matrices $L$ and $R$ that approximate the Fisher information or Hessian along row and column dimensions of weight tensors. 
- **Update rule**: 
  $$W \leftarrow W - \eta L^{-1/2} G R^{-1/2}$$
  where $G$ is the gradient, $L = \sum_t G_t G_t^T$, and $R = \sum_t G_t^T G_t$ (with exponential averaging and regularization).
- **Why "whitening"**: This transformation decorrelates the gradient components, making updates isotropic: $\text{Cov}(\text{whitened } G) \approx I$. Treats all directions in weight space equally, unlike raw gradients which can be dominated by a few directions due to ill-conditioning.
- **Computational efficiency**: Key innovation is using Newton-Schulz iteration to compute $M^{-1/2}$ on GPUs: starting from $Z_0 = M / \|M\|$, iterate $Z_{k+1} = \frac{1}{2}Z_k(3I - Z_k^2)$. Converges quadratically without expensive eigendecomposition.
- **Concrete results**: On MLPerf ResNet-50, Shampoo achieved target accuracy with ~20% fewer training steps than SGD and ~41% less total wall-clock time after amortizing computational overhead. Scaled to production systems (e.g., Google Ads CTR prediction) by distributing matrix operations to CPU workers.
- **Connection to Muon**: Muon simplifies Shampoo by removing preconditioner accumulation ($L$ and $R$), directly orthogonalizing the gradient/momentum. This maintains the isotropy benefits with simpler computations and no second-moment storage.

</details>

<details>
<summary><b>Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner</b> — <i>Luke Jasper Latham, Rene Vidal</i></summary>

[Paper Link](https://arxiv.org/abs/2308.09627)

- **Decomposition result**: Shows Shampoo's preconditioner $L^{-1/2} G R^{-1/2}$ can be decomposed into two components:
  1. **Spectral normalization**: Normalizes the spectral norm of the gradient (largest singular value → 1)
  2. **Variance adaptation**: Adaptively rescales components based on second-moment statistics (like Adam's per-parameter adaptation)
- **Eigenvalue spread control**: The preconditioner specifically controls the spread of gradient-covariance eigenvalues. If raw gradients have eigenvalues $[\lambda_1, \lambda_2, ..., \lambda_n]$ with $\lambda_1 \gg \lambda_n$ (high condition number), the preconditioner brings them closer together, reducing the condition number of the optimization landscape.
- **Why this helps**: A benign optimization regime with lower condition number means:
  - More stable training (less sensitive to learning rate)
  - Faster convergence (fewer oscillations)
  - Better generalization (less overfitting to high-curvature directions)
- **Relevance to Muon**: Muon inherits Shampoo's spectral normalization component but drops the variance adaptation component. This decomposition helps understand which aspects of matrix-whitening are essential for Muon's benefits.

</details>

---

## 📄 Original Literature

> Foundational papers introducing and deriving Muon

### 🎯 Core Papers

<details>
<summary><b>Muon: An optimizer for hidden layers in neural networks</b> — <i>Keller Jordan (OpenAI)</i></summary>

[Blog Post](https://kellerjordan.github.io/posts/muon/)

**Core algorithm**: Applies momentum, then orthogonalizes the momentum update before stepping. For gradient $G$ with SVD $G = U\Sigma V^T$, Muon replaces it with $\tilde{G} = UV^T$, setting all singular values to 1. This creates an orthonormal update with equal energy in every direction and bounded spectral norm.

**Newton-Schulz orthogonalization**: Efficiently approximates $G(G^T G)^{-1/2}$ through iterative computation:
$$Z_{k+1} = \frac{3}{2}Z_k - \frac{1}{2}Z_k^3$$
starting from $Z_0 = G/\|G\|$. Avoids expensive SVD computation while achieving same effect on accelerators. Converges cubically.

**Hybrid approach**: Uses Muon for all 2D weight matrices (where matrix structure is meaningful), but AdamW for scalar parameters (biases, LayerNorm gains, embedding scaling) and often first/last layers for stability.

**Relationship to prior work**:
- **Shampoo**: Muon simplifies Shampoo by removing preconditioner accumulation, arriving at direct gradient orthogonalization
- **Orthogonal-SGDM**: Muon applies momentum *before* orthogonalization (vs. after) and uses Newton-Schulz (vs. expensive SVD)

**Concrete results**: 
- CIFAR-10: 94% accuracy in 20% less time than previous optimizers
- NanoGPT: Set new records on language modeling speedruns, reaching target validation loss with fewer tokens

**Key design choice**: Decoupled weight decay (like AdamW) applied separately from the orthogonalized update—crucial for proper implicit regularization.

</details>

<details>
<summary><b>Deriving Muon</b> — <i>Jeremy Bernstein (MIT)</i></summary>

[Blog Post](https://jeremybernste.in/writing/deriving-muon)

**Steepest descent framework**: Shows that many optimizers (including Muon) can be viewed as steepest descent under different norm constraints. The update is: $w_{t+1} = w_t - \eta \cdot \arg\min_{\|\Delta\|_{\mathcal{K}} \leq 1} \langle \nabla f(w_t), \Delta \rangle$, where $\|\cdot\|_{\mathcal{K}}$ is a norm induced by convex function $\mathcal{K}$.

**Lion-K family**: Generalizes the Lion optimizer (Chen et al., 2023) to arbitrary convex functions $\mathcal{K}$. Lion uses $\mathcal{K}(g) = \|g\|_1$ (L1 norm), yielding sign-based updates.

**Muon's position**: Muon corresponds to Lion-K with $\mathcal{K}(G) = \|G\|_*$ (nuclear norm = sum of singular values). The "sign function" in this case becomes the matrix sign function, which for $G = U\Sigma V^T$ is $\text{sign}(G) = UV^T$ —exactly Muon's orthogonalized update.

**Mathematical connection**: For nuclear norm, the proximal operator gives: $\text{prox}_{\mathcal{K}}(G) = G(G^T G)^{-1/2}$, which Muon approximates via Newton-Schulz.

**Implication**: This unifying view shows Muon is principled (not ad-hoc) and opens design space for other matrix-based optimizers by varying $\mathcal{K}$.

</details>

### 📚 Related Theoretical Foundations

| Paper | Author(s) | Link | Key Insight |
|-------|-----------|------|-------------|
| **Modular Manifolds** | Jeremy Bernstein | [Link](https://thinkingmachines.ai/blog/modular-manifolds/) | Establishes geometric/manifold perspective underlying Muon's orthogonalization |
| **Old Optimizer, New Norm: An Anthology** | Jeremy Bernstein, Laker Newhouse | [Link](https://arxiv.org/abs/2409.20325) | Historical survey connecting classical methods to modern norm-based perspectives |
| **Scalable Optimization in the Modular Norm** | Tim Large et al. (MIT) | [Link](https://arxiv.org/abs/2405.14813) | Develops scalable algorithms for optimization under modular norm constraints |
| **Duality, Weight Decay, and Metrized Deep Learning** | Laker Newhouse | [Link](https://www.lakernewhouse.com/thesis.pdf) | PhD thesis on weight decay, duality theory, and metric-based optimization |
| **Understanding Muon Chapter 1: Into the Matrix** | Laker Newhouse | [Link](https://www.lakernewhouse.com/writing/muon-1) | Educational deep-dive into Muon's matrix operations and Newton-Schulz iteration |
| **Depths of First-Order Optimization** | Jeremy Bernstein | [Link](https://docs.google.com/presentation/d/1PIAChMGGwhmdUxDPyOo1o8Qlhq3h_ofV2mhBb6JHH04) | Presentation on theoretical depths of first-order methods |

---

## 🧮 Theoretical Analysis

> Convergence properties and optimization-theoretic characterizations of Muon

<details>
<summary><b>On the Convergence Analysis of Muon</b> — <i>Wei Shen et al. (UVA, UBC, Meta, UW-Madison)</i></summary>

[Paper Link](https://arxiv.org/abs/2505.23737)

**Main result**: Provides convergence rate analysis comparing Muon against Gradient Descent (GD). Establishes conditions under which Muon outperforms GD.

**When Muon wins**: Shows Muon benefits from two structural properties common in neural networks:
1. **Low-rank Hessians**: When $H \approx \sum_{i=1}^r \lambda_i v_i v_i^T$ with $r \ll d$ (rank $r$ much smaller than dimension $d$)
2. **Approximate block-diagonal structure**: When Hessian is approximately block-diagonal (different parameter groups have weak cross-interactions)

**Convergence rate**: Under smoothness and PL conditions, Muon achieves $\mathcal{O}(1/T)$ convergence rate (same order as GD), but with better constants when above structures hold. The improvement factor scales with the degree of low-rankness and block-diagonality.

**Experimental validation**: Empirical results on neural network training confirm theoretical predictions—Muon's advantage grows with network depth (where these structures become more pronounced).

</details>

<details>
<summary><b>A Note on the Convergence of Muon</b> — <i>Jiaxiang Li, Mingyi Hong (University of Minnesota)</i></summary>

[Paper Link](https://arxiv.org/abs/2502.02900)

**Alternative perspective**: Provides convergence analysis under different assumptions than the previous paper, potentially covering different problem classes or relaxing certain conditions.

**Complementary techniques**: Uses different proof techniques that may provide tighter bounds in specific regimes or offer insights into different aspects of Muon's behavior.

</details>

<details>
<summary><b>Muon Optimizes Under Spectral Norm Constraints</b> — <i>Lizhang Chen, Jonathan Li, Qiang Liu (UT Austin)</i></summary>

[Paper Link](https://arxiv.org/abs/2506.15054)

**Theoretical framework**: Places Muon within the Lion-K family of optimizers, showing Muon corresponds to Lion-K equipped with the nuclear norm (K is sum of singular values).

**Core result**: Proves that Muon with decoupled weight decay implicitly solves the constrained optimization problem: min f(W) subject to ||W||_σ ≤ C, where ||W||_σ is the spectral norm (largest singular value) and C is a constant determined by the weight decay coefficient.

**Mechanism**: The orthogonalization step acts as a projection onto the constraint manifold {W: ||W||_σ = C}, similar to projected gradient descent. Each update doesn't increase the spectral norm—it stays within the spectral norm ball.

**Implications**:
1. Controls model capacity through spectral norm bounds (related to Lipschitz constant)
2. Prevents weight explosion naturally
3. Improves generalization via implicit spectral regularization
4. Opens design space: varying convex function K yields broader class of constrained optimizers

**Empirical validation**: Experiments on ResNet and LLaMA architectures confirm Muon reduces overfitting and maintains robust training dynamics.

</details>

---

## 🔬 Understanding Property

> How Muon's spectral design affects learning dynamics and feature acquisition

<details>
<summary><b>Muon Outperforms Adam in Tail-End Associative Memory Learning</b> — <i>Shuche Wang et al. (NUS, UMN, Sea AI Lab, Yale)</i></summary>

[Paper Link](https://arxiv.org/abs/2509.26030)

**Key insight**: Shows Muon enhances isotropy of weight matrices, leading to more balanced knowledge acquisition. Particularly effective at learning tail-end (rare/infrequent) associations in large language models compared to Adam, which tends to prioritize frequent patterns at the expense of rare ones.

</details>

<details>
<summary><b>How Muon's Spectral Design Benefits Generalization: A Study on Imbalanced Data</b> — <i>Bhavya Vasudeva et al. (USC, UBC)</i></summary>

[Paper Link](https://arxiv.org/abs/2510.22980)

**Core finding**: Standard GD learns principal components sequentially (dominant first), while Muon/Shampoo learn all components at similar rates, creating more balanced feature learning.

**Theoretical model**: Introduces idealized Spectral Gradient Descent (SpecGD) that computes G = UΣV^T and updates with UV^T (equalized singular values). Proves in Gaussian-mixture classification with imbalance, GD prioritizes top principal component while SpecGD learns all components simultaneously. Effect amplifies with network depth.

**Concrete improvements**: On vision datasets with class imbalance (e.g., 100:1 ratio), Muon achieved over 5% higher balanced accuracy than AdamW. This gap appears early in training and persists, showing Muon learns minority-class features from the start rather than late in training.

**Why adaptive LR isn't enough**: Giving Adam per-layer adaptive step sizes (normalizing each layer's gradient norm) does NOT eliminate the gap—the advantage is intrinsic to the orthogonalized update direction, not just learning rate magnitude.

**Practical impact**: Provides straightforward way to improve minority-class performance without complex data re-sampling or loss re-weighting schemes. May sacrifice tiny amount of majority-class accuracy but overall test accuracy improves when imbalance is significant.

</details>

<!-- > [!IMPORTANT]
> **Featured Paper**: This paper provides crucial theoretical insights into Muon's implicit bias and why it learns more balanced classifiers. -->

<details open>
<summary><b>⭐ Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data</b> — <i>Chen Fan, Mark Schmidt, Christos Thrampoulidis (UBC)</i></summary>

[Paper Link](https://arxiv.org/abs/2502.04664)

**TL;DR:** Muon and spectral-type optimizers have an internal bias that automatically pushes them toward solutions that are low-rank and spectrally balanced — even without any explicit regularization.

---

**Problem setting**: Multiclass classification on linearly separable data, where multiple separating hyperplanes exist. The implicit bias question: which separator does each optimizer converge to in the limit of gradient flow?

* When an optimizer runs for a very long time and the training loss goes all the way to zero, what exact classifier (or separating boundary) does it end up with?
*  If multiple different weight matrices can fit the data perfectly, which one does an optimizer like Muon pick, and what does that reveal about its hidden bias?
* What is the implicit geometric bias of spectral-norm optimizers like Muon — and can we rigorously prove which kind of margin or simplicity they maximize?

**Classical result (GD)**: GD on logistic/cross-entropy loss converges to the max-$\ell_2$ margin solution: the separator $w^*$ that maximizes $\min_i \frac{y_i \langle w, x_i \rangle}{\|w\|_2}$ (maximum minimum margin in Euclidean norm).

> **Intuition:** When you train a linear classifier with ordinary gradient descent (say, logistic or cross-entropy loss), the algorithm keeps adjusting the weights to separate the data more confidently. As the loss approaches zero, the direction of the weights stops changing—it converges to one specific separator that maximizes the minimum distance between data points and the decision boundary, measured using the ordinary Euclidean ($L_2$) notion of length. This is why "Gradient descent implicitly finds the $L_2$ max-margin classifier."

**Spectral Descent (and Muon) characterization**: The paper analyzes Spectral Gradient Descent (SpecGD), which orthogonalizes gradients before stepping (equivalent to setting all singular values to 1). For a gradient matrix $G$ with classes along columns, SpecGD uses update direction $\tilde{G} = G(G^T G)^{-1/2}$.

> **Intuition:** Spectral Descent looks at the weight matrix as a geometric object. Before taking a step, it orthogonalizes the gradient, removing scaling differences along different directions and keeping only the shape of the update. Mathematically, that's like replacing the gradient matrix $G$ with $UV^T$, where $G =U \Sigma V$ is its singular-value decomposition. Spectral Descent (and Muon) move in a way that respects the "spectral geometry" — they pay attention to how correlated different classes and features are, not to the raw coordinate values.

**Main theoretical result**: Shows that spectral methods (including Muon) converge to a different implicit bias than standard GD. Rather than maximizing margin in $\ell_2$ norm, they implicitly optimize a margin criterion that accounts for the spectral structure of the data.

> All of this means Muon and Spectral Descent don't just optimize faster — they're actually aiming for a different kind of simplicity. Instead of minimizing "overall size" of the weights (like Euclidean norm), they minimize the "strongest directional power" of the matrix (its spectral norm).

**Multiclass specifics**: In the multiclass setting with $K$ classes, the solution space is $\mathbb{R}^{d \times K}$ (weight matrix $W$ with one column per class). The paper characterizes:

1. **Direction of convergence**: What directional properties the limit $W^* / \|W^*\|$ satisfies
2. **Margin type**: What notion of margin is implicitly maximized (spectral norm-based vs. Frobenius norm-based vs. nuclear norm-based)
3. **Class balance**: How the method treats different classes (balanced vs. biased toward majority)

**Comparison to GD**: Key differences in the implicit bias:
- **GD**: Converges to max-$\ell_2$ margin, can be biased toward majority classes
- **Spectral methods (Muon/SpecGD)**: More balanced treatment of classes due to orthogonalization equalizing contribution from each class's gradient. The margin criterion involves spectral norm $\|W\|_{\sigma} = \sigma_{\max}(W)$ rather than Frobenius norm.

**Experimental results:**

![Implicit Bias Experiments](implicit%20bias_exps.png)

The x-axis = number of training iterations. The y-axis = how close the model's current margin is to the maximum possible margin under different norms.

| Panel | Optimizer | Which margin increases the most? | Meaning |
|:------|:-----------|:--------------------------------|:---------|
| (a) SignGD | Max-norm (blue) | The blue curve rises and dominates | SignGD naturally prefers the **L∞ margin** — consistent with its L∞ geometry. |
| (b) NGD | L₂ (orange) | The orange curve rises and dominates | Gradient descent converges to the **L₂ max-margin separator** — the classical result. |
| (c) Spectral-GD | Spectral (green) | The green curve rises highest | Spectral Descent converges to the **spectral-norm margin** — i.e., the smallest dominant singular value. |
| (d) Muon | Spectral (green) | Same as Spectral-GD | Muon behaves just like Spectral Descent — it maximizes the **spectral margin**. |

**Practical implications**: 
1. On imbalanced multiclass data, Muon's implicit bias leads to better minority-class separation
2. The solution is more "democratic" across classes—no single class dominates the parameter updates
3. Connects to the imbalanced data results: the implicit bias explains *why* Muon learns minority classes better

**Connection to max-margin theory**: Extends classical results on implicit bias of GD (Soudry et al., 2018) to spectral/matrix-structured optimizers, providing theoretical foundation for understanding Muon's behavior on classification tasks.

</details>

---



## ⚖️ Critical Batch Size

> Understanding optimal batch size scaling for Muon in large-scale training

<details>
<summary><b>Optimal Scaling Needs Optimal Norm</b> — <i>Oleg Filatov et al. (Julich Supercomputing Centre)</i></summary>

[Paper Link](https://arxiv.org/abs/2510.03871)

**Key insight**: Investigates the relationship between optimizer norms and optimal scaling behavior. Shows that different norms (including spectral norm used by Muon) affect the critical batch size and scaling efficiency differently.

</details>

<details>
<summary><b>Convergence Bound and Critical Batch Size of Muon Optimizer</b> — <i>Naoki Sato et al. (Meiji, Mila, Université de Montréal)</i></summary>

[Paper Link](https://arxiv.org/abs/2507.01598)

**Four variants analyzed**: Provides convergence proofs for Muon in four practical settings:
1. Base Muon (no momentum, no weight decay)
2. Muon + Nesterov momentum
3. Muon + weight decay
4. Muon + both Nesterov momentum and weight decay (standard configuration)

**Weight decay tightens bounds**: Shows that adding weight decay yields strictly tighter theoretical convergence bounds. The interplay between weight decay coefficient $\lambda$ and learning rate $\eta$ is clarified—optimal $\lambda$ scales with $1/\eta$.

**Critical batch size**: Derives Muon's critical batch size $B_{\text{crit}}$ that minimizes computational cost: the point beyond which increasing batch size gives diminishing returns in wall-clock time. Formula involves gradient noise scale and problem-specific constants.

**Practical guidance**: The analysis identifies which hyperparameters govern critical batch size, helping practitioners choose optimal batch sizes for their hardware and problem scale.

**Experimental validation**: Theoretical predictions validated through experiments, confirming practical utility of the bounds.

</details>

---

## 📊 Empirical Evaluation

> Comprehensive benchmarks and empirical studies comparing Muon to other optimizers

<details>
<summary><b>Practical Efficiency of Muon for Pretraining</b> — <i>Essential AI (Ishaan Shah, Anthony M. Polloreno, Karl Stratos, et al.)</i></summary>

[Paper Link](https://arxiv.org/abs/2505.02222)

**Main claim**: Muon expands the Pareto frontier over AdamW on the compute-time tradeoff. For a given target loss, Muon reaches it faster in wall-clock time, or given a fixed time budget, Muon reaches lower loss.

**Large batch efficiency**: Key advantage is maintaining data efficiency at large batch sizes, far beyond the typical critical batch size. While AdamW's performance degrades with very large batches, Muon continues to scale effectively, enabling faster training on multi-GPU systems.

**Combination with muP**: Studies integration of Muon with maximal update parameterization (muP) for efficient hyperparameter transfer across model scales. Proposes telescoping algorithm that accounts for all error sources in muP scaling with modest computational overhead.

**Experimental scope**: Validates findings on models up to 4 billion parameters, with ablations on data distribution and architecture choices.

**Practical impact**: Enables more economical training by better utilizing large batch sizes without sacrificing final performance.

</details>

<details>
<summary><b>Muon is Scalable for LLM Training</b> — <i>Moonshot AI (Kimi2), UCLA</i></summary>

[Paper Link](https://arxiv.org/abs/2502.16982)

**Scaling challenges addressed**: Identifies two critical techniques for scaling Muon: (1) adding decoupled weight decay (crucial for implicit regularization), and (2) carefully adjusting per-parameter update scales to balance magnitude across layers.

**Moonlight implementation**: Trained 3B and 16B-parameter Mixture-of-Experts models with 5.7T tokens using Muon, working out-of-the-box without extensive hyperparameter tuning.

**Concrete efficiency gains**: Achieves approximately 2X computational efficiency over AdamW in compute-optimal training—reaching the same validation loss with roughly half the FLOPs.

**Pareto frontier improvement**: The 16B MoE model achieves better perplexity for a given compute budget than prior models, demonstrating Muon improves not just speed but also the fundamental compute-performance tradeoff.

**Practical impact**: Shows Muon is production-ready for billion-scale models, with comparable hyperparameter robustness to AdamW.

</details>

<details>
<summary><b>Muon: Training and Trade-offs with Latent Attention and MoE</b> — <i>Sushant Mehta, Raj Dandekar, Rajat Dandekar, Sreedath Panat</i></summary>

[Paper Link](https://arxiv.org/abs/2509.24406)

**Theoretical contributions**: Rigorous analysis including convergence rates, spectral regularization properties preventing gradient explosion, connection to natural gradient descent on the Stiefel manifold, and equivalence to steepest gradient descent under spectral norm.

**Efficiency on transformers (30M-200M params)**: Muon reaches target loss with 48-52% of the training computation required by AdamW while maintaining or improving final perplexity.

**Synergy with modern architectures**: When combined with Multi-Head Latent Attention (MLA) and Mixture-of-Experts (MoE):
- 68% memory reduction
- 3.2× inference speedup
- 8-12% perplexity improvement

**Practical significance**: Demonstrates Muon is particularly effective when paired with efficient architectures, offering multiplicative gains rather than just additive improvements.

</details>

### 📈 Benchmark Studies

| Paper | Author(s) | Link | Key Finding |
|-------|-----------|------|-------------|
| **Fantastic Pretraining Optimizers and Where to Find Them** | Kaiyue Wen et al. (Stanford) | [Link](https://arxiv.org/abs/2509.02046) | Systematic benchmark across architectures and training scales |
| **Benchmarking Optimizers for Large Language Model Pretraining** | Andrei Semenov et al. (EPFL) | [Link](https://arxiv.org/abs/2509.01440) | Comprehensive benchmark up to 1.3B parameters |
| **Optimization Benchmark for Diffusion Models on Dynamical Systems** | Fabian Schaipp (Inria) | [Link](https://arxiv.org/abs/2510.19376v1) | Extends evaluation to diffusion models |
| **The Potential of Second-Order Optimization for LLMs** | Natalie Abreu et al. (Harvard) | [Link](https://arxiv.org/abs/2510.09378) | Context for Muon's position vs second-order methods |
| **What Really Matters in Matrix-Whitening Optimizers?** | Kevin Frans et al. (UC Berkeley) | [Link](https://arxiv.org/abs/2510.25000) | Analyzes essential components of matrix-whitening optimizers |

---

## ⚡ Efficient Algorithm

> Practical improvements reducing Muon's computational and memory costs

<details>
<summary><b>LiMuon: Light and Fast Muon Optimizer for Large Models</b> — <i>Feihu Huang et al. (Nanjing U of Aeronautics and Astronautics)</i></summary>

[Paper Link](https://arxiv.org/abs/2509.14562)

**Variance reduction enhancement**: Adds momentum-based variance reduction to Muon's orthogonalization step, improving convergence rates especially in stochastic settings with high gradient noise.

**Memory-computation tradeoff**: Discusses two variants:
1. **Full SVD**: Exact orthogonalization, higher cost but precise
2. **Randomized SVD (RSVD)**: Approximate orthogonalization with $\tilde{O}(k)$ complexity where $k \ll \min(m,n)$ for $m \times n$ matrix. Trades slight accuracy for major speedup.

**Convergence improvement**: Theoretical analysis shows LiMuon achieves faster convergence rate than vanilla Muon under standard assumptions.

**Scalability**: Memory-efficient variants enable application to models where full Muon would be prohibitive.

</details>

<details>
<summary><b>Effective Quantization of Muon Optimizer States</b> — <i>Aman Gupta et al. (Mubank, LinkedIn)</i></summary>

[Paper Link](https://arxiv.org/abs/2509.23106)

**Key insight**: Demonstrates that Muon's optimizer states can be effectively quantized without significant performance loss. Reduces memory footprint, making Muon more practical for memory-constrained environments and enabling larger models to be trained.

</details>

<details>
<summary><b>NorMuon: Making Muon more efficient and scalable</b> — <i>Zichong Li et al. (Georgia Tech, Microsoft AI)</i></summary>

[Paper Link](https://arxiv.org/abs/2510.05491)

**Problem identified**: While Muon's orthogonalization reduces condition numbers, it leads to non-uniform neuron norms post-orthogonalization, causing certain neurons to dominate the optimization process.

**Solution**: NorMuon maintains second-order momentum statistics for each neuron and applies row-wise normalization after orthogonalization, ensuring balanced parameter utilization while preserving Muon's conditioning benefits.

**Concrete improvements**: In 1.1B parameter pretraining, NorMuon achieves 21.74% better training efficiency than Adam and 11.31% improvement over vanilla Muon, while maintaining comparable memory footprint to Muon.

**Distributed implementation**: Presents efficient implementation under FSDP2 framework, distributing orthogonalization computations across devices for scalability.

**Key insight**: Demonstrates that orthogonalization and adaptive learning rates are complementary techniques, opening new optimizer design directions.

</details>

---

## 🌐 Distributed Setting

> Adapting Muon for distributed and federated training scenarios

| Paper | Author(s) | Link | Key Contribution |
|-------|-----------|------|------------------|
| **Dion: Distributed Orthonormalized Updates** | Kwangjun Ahn et al. (Microsoft Research, Harvard) | [Link](https://arxiv.org/abs/2504.05295) | Extends orthogonalization to distributed training with efficient communication |
| **MuLoCo: Muon is a practical inner optimizer for DiLoCo** | Benjamin Thérien et al. (Mila, UdM, Concordia) | [Link](https://arxiv.org/abs/2505.23725) | Muon as inner optimizer for distributed low-communication training |
| **MuonBP: Faster Muon via Block-Periodic Orthogonalization** | Ahmed Khaled et al. (Princeton, AWS, UMN) | [Link](https://arxiv.org/abs/2510.16981) | Block-periodic orthogonalization reduces computational overhead |

---

## 📈 Scaling

> Strategies for scaling second-order and matrix-based optimizers to very large models

| Paper | Author(s) | Link | Key Insight |
|-------|-----------|------|-------------|
| **How to Scale Second-Order Optimization** | Zixi (Charlie) Chen et al. (NYU) | [Link](https://openreview.net/pdf/d9ff9b9df54dd1e155b0d792f9a86d879a81a53c.pdf) | Practical strategies for scaling second-order methods to large neural networks |

---

## 🛡️ Regularization

> Regularization techniques compatible with and enhancing Muon's implicit bias

| Paper | Author(s) | Link | Key Insight |
|-------|-----------|------|-------------|
| **Cautious Weight Decay** | Lizhang Chen et al. (UT Austin, Google) | [Link](https://arxiv.org/abs/2510.12402) | Adaptive weight decay strategy particularly relevant for spectrum-aware optimizers |

---

## ✨ Enhancement

> Extensions and complementary techniques that can be combined with Muon

| Paper | Author(s) | Link | Key Contribution |
|-------|-----------|------|------------------|
| **MARS: Unleashing the Power of Variance Reduction for Training Large Models** | Huizhuo Yuan et al. (ByteDance, UCLA) | [Link](https://arxiv.org/abs/2411.10438) | Variance reduction + adaptive learning rates |
| **Training Deep Learning Models with Norm-Constrained LMOs** | Thomas Pethick et al. (EPFL, U Paris-Saclay) | [Link](https://arxiv.org/abs/2502.07529) | SCION with norm-constrained LMOs |
| **REG: A Regularization Optimizer for Robust Training Dynamics** | Zehua Liu et al. (Huawei Noah's Ark Lab) | [Link](https://arxiv.org/abs/2510.03691) | Explicit regularization schemes |
| **Noise-Adaptive Layerwise Learning Rates** | Jie Hao et al. (George Mason) | [Link](https://arxiv.org/abs/2510.14009) | LANTON with noise-adaptive layer-wise learning rates |

---

## 📝 Blog Posts

> Accessible explanations and practical insights from practitioners

| Blog Post | Author | Link | Focus |
|-----------|--------|------|-------|
| **Deep Learning Optimizers as Steepest Descent in Normed Spaces** | Franz Louis Cesista | [Link](https://leloykun.github.io/ponder/steepest-descent-opt/) | Theoretical framework made accessible |
| **Muon and a Selective Survey on Steepest Descent** | Franz Louis Cesista | [Link](https://leloykun.github.io/ponder/steepest-descent-non-riemannian/) | Geometric perspective on Muon |
| **Squeezing 1-2% Efficiency Gains Out of Muon** | Franz Louis Cesista | [Link](https://leloykun.github.io/ponder/muon-opt-coeffs/) | Optimizing Newton-Schulz coefficients |

---
  
## 🔗 Related Work and Perspectives

> Papers providing broader context, alternative approaches, or complementary insights

<details>
<summary><b>Muon Optimizer Accelerates Grokking</b> — <i>Amund Tveit et al. (Microsoft Norway)</i></summary>

[Paper Link](https://arxiv.org/abs/2504.16041)

**What is grokking**: A phenomenon where models exhibit sudden delayed generalization—continuing to train past zero training loss eventually leads to dramatic improvement in test accuracy.

**Experimental setup**: Seven numerical tasks using modern Transformer architecture, systematically varying optimizer (Muon vs. AdamW) and softmax activation function variants.

**Concrete results**: Muon reduced mean grokking epoch from 153.09 to 102.89 across all configurations—a statistically significant 33% reduction (t = 5.0175, p = 6.33e-08). This demonstrates Muon's spectral norm constraints and second-order information facilitate faster transition from memorization to true generalization.

**Why it matters**: Suggests that optimizer choice fundamentally affects learning dynamics beyond just convergence speed, influencing when and how models develop genuine understanding vs. mere pattern memorization.

</details>

### 🔬 Theoretical Perspectives

| Paper | Author(s) | Link | Key Contribution |
|-------|-----------|------|------------------|
| **Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization** | Dmitry Kovalev (Yandex Research) | [Link](https://arxiv.org/abs/2503.12645) | Trust-region optimization perspective on gradient orthogonalization |
| **PolarGrad: A Class of Matrix-Gradient Optimizers** | Tim Tsz-Kit Lau et al. (U Penn) | [Link](https://arxiv.org/abs/2505.21799) | Unifying preconditioning perspective for matrix-gradient methods |
| **The Polar Express: Optimal Matrix Sign Methods** | Noah Amsel et al. (NYU, Flatiron Institute) | [Link](https://arxiv.org/abs/2505.16932) | Optimal methods for computing matrix sign function |
| **Towards understanding of orthogonalization in Muon** | Valentyn Boreiko et al. (U Tübingen, Amazon) | [Link](https://openreview.net/forum?id=4vzhqq5hpX) | Workshop paper on orthogonalization mechanisms (ICML 2025) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For adding new papers, please maintain the existing format and ensure all links are working.

## 📜 License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the contributors have waived all copyright and related or neighboring rights to this work.
