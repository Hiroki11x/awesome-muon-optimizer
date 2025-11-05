# Awesome Muon Optimizer [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of research on Muon, a spectrum-aware optimizer that orthogonalizes gradient updates for neural network training. This collection documents the theoretical foundations, empirical evaluations, and practical developments around Muon and related spectral/matrix-based optimization methods.

## Background: Spectral Bias and Adaptive Optimizers

Understanding the motivation for spectrum-aware optimizers like Muon.

- **Spectral Bias in Practice: the Role of Function Frequency in Generalization**
  - https://papers.neurips.cc/paper_files/paper/2022/file/306264db5698839230be3642aafc849c-Paper-Conference.pdf
  - **Key insight**: Establishes that neural networks learn low-frequency (smooth) patterns before high-frequency (detailed) ones during gradient descent. High-frequency components are learned more slowly. This spectral bias aids generalization on balanced data by promoting smoother functions, but can neglect rare features or minority classes in imbalanced settings. Motivates development of spectrum-aware optimizers.

- **Adam vs. SGD: Achieving Comparable Generalization in Image Classification Through Adaptive Techniques**
  - https://rjpn.org/ijcspub/viewpaperforall.php?paper=IJCSP25B1033
  - **Key insight**: Shows the generalization gap between Adam and SGD in computer vision can be closed with appropriate techniques (AdamW with decoupled weight decay, layer-wise normalization). Demonstrates that adaptive optimizers' poor generalization was not fundamental but an artifact of missing regularization. AdamW with proper tuning can match SGD's generalization.

- **Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models**
  - https://arxiv.org/abs/2210.14199
  - **Key insight**: Reveals that models with identical pre-training loss can have vastly different downstream task performance depending on optimizer and training dynamics. Shows ~6% downstream accuracy gap between adversarial and standard training with same loss. Links optimizer choice to solution flatness (Hessian trace): flatter solutions transfer better. Demonstrates optimizer's implicit bias affects feature quality beyond just loss minimization.

- **Deconstructing What Makes a Good Optimizer for Language Models**
  - https://arxiv.org/abs/2407.07972
  - **Key insight**: Large-scale study (up to 1.3B parameters) comparing AdamW, AdaFactor, Lion, and SignSGD on transformers. Finds these optimizers reach similar final perplexity and zero-shot accuracy when well-tuned, but differ in hyperparameter robustness. Plain SGD fails on transformers without heavy modification. Suggests optimizer choice affects solution nature and downstream performance more than pre-training metrics.

- **Scalable Second Order Optimization for Deep Learning (Shampoo)**
  - Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
  - https://arxiv.org/abs/2002.09018
  - **Key insight**: Introduces Shampoo, which preconditions gradients using L^{-1/2} G R^{-1/2} where L and R approximate Fisher/Hessian. "Whitens" gradients to make updates isotropic. On MLPerf ResNet-50, achieved ~20% fewer steps and ~41% less wall-clock time than SGD. Uses Newton-Schulz iteration for efficient matrix roots on accelerators. Provides foundation for understanding matrix-based preconditioning that influenced Muon's design.

- **Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner**
  - Luke Jasper Latham, Rene Vidal
  - https://arxiv.org/abs/2308.09627
  - **Key insight**: Decomposes Shampoo's preconditioner into spectral normalization and variance adaptation components. Shows how preconditioner controls gradient-covariance eigenvalue spread to maintain benign optimization regime. Helps understand why matrix-whitening methods improve generalization and provides insights applicable to Muon's design.

## Original Literature

### Core Papers

- **Muon: An optimizer for hidden layers in neural networks**
  - Keller Jordan (OpenAI)
  - https://kellerjordan.github.io/posts/muon/
  - **Key insight**: Introduces Muon optimizer which orthogonalizes momentum updates using Newton-Schulz iteration. Replaces gradient G = UΣV^T with G̃ = UV^T (making all singular values equal to 1), creating bounded spectral norm updates. Achieved 20% faster CIFAR-10 training (94% accuracy) and set records on NanoGPT speedruns. Uses AdamW for scalar parameters (biases, LayerNorm) and Muon for weight matrices.

- **Deriving Muon**
  - Jeremy Bernstein (MIT)
  - https://jeremybernste.in/writing/deriving-muon
  - **Key insight**: Provides theoretical derivation showing Muon fits within the steepest descent framework under specific norm constraints. Connects to the broader Lion-K framework where K is the nuclear norm (sum of singular values).
 
- **Modular Manifolds**
  - Jeremy Bernstein (Thinking Machine)
  - https://thinkingmachines.ai/blog/modular-manifolds/
  - **Key insight**: Establishes the geometric/manifold perspective underlying Muon's orthogonalization approach, viewing optimization through modular norm constraints.

- **Old Optimizer, New Norm: An Anthology**
  - Jeremy Bernstein, Laker Newhouse (MIT)
  - https://arxiv.org/abs/2409.20325
  - **Key insight**: Historical and theoretical survey connecting classical optimization methods to modern norm-based perspectives, providing context for Muon's design principles.
 
- **Scalable Optimization in the Modular Norm**
  - Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, Jeremy Bernstein (MIT)
  - https://arxiv.org/abs/2405.14813
  - **Key insight**: Develops scalable algorithms for optimization under modular norm constraints, laying groundwork for practical implementation of methods like Muon at scale.

- **Duality, Weight Decay, and Metrized Deep Learning**
  - Laker Newhouse (MIT)
  - https://www.lakernewhouse.com/thesis.pdf
  - **Key insight**: PhD thesis exploring the relationship between weight decay, duality theory, and metric-based deep learning optimization. Shows how decoupled weight decay (as in AdamW) connects to implicit constraint formulations.
 
- **Understanding Muon Chapter 1: Into the Matrix**
  - Laker Newhouse (MIT)
  - https://www.lakernewhouse.com/writing/muon-1
  - **Key insight**: Educational deep-dive into Muon's matrix operations and the Newton-Schulz iteration used for efficient orthogonalization.
 
- **Depths of First-Order Optimization**
  - Jeremy Bernstein (MIT)
  - https://docs.google.com/presentation/d/1PIAChMGGwhmdUxDPyOo1o8Qlhq3h_ofV2mhBb6JHH04
  - **Key insight**: Presentation exploring the theoretical depths of first-order methods and their connection to geometry-aware optimization.

## Theoretical Analysis

Convergence properties and optimization-theoretic characterizations of Muon.

- **On the Convergence Analysis of Muon**
  - Da Chang, Yongxiang Liu, Ganzhao Yuan (UVA, UBC, Meta, UW-Madison)
  - https://arxiv.org/abs/2505.23737
  - **Key insight**: Provides formal convergence analysis of Muon, establishing rates and conditions under which it converges to stationary points. Part of establishing Muon's theoretical soundness.

- **A Note on the Convergence of Muon**
  - Jiaxiang Li, Mingyi Hong (University of Minnesota)
  - https://arxiv.org/abs/2502.02900
  - **Key insight**: Alternative convergence analysis with different assumptions or proof techniques, complementing the previous convergence work.

- **Muon Optimizes Under Spectral Norm Constraints**
  - Lizhang Chen, Jonathan Li, Qiang Liu (UT Austin)
  - https://arxiv.org/abs/2506.15054
  - **Key insight**: Shows that Muon's orthogonalization step implicitly solves min f(W) subject to ||W||_σ ≤ C (spectral norm constraint). Provides Lagrangian perspective: Muon performs implicit spectral regularization per layer, keeping weights within bounded spectral-norm regions. This controls model capacity, improves generalization, and stabilizes training by preventing weight explosion.

## Understanding Property

How Muon's spectral design affects learning dynamics and feature acquisition.

- **Muon Outperforms Adam in Tail-End Associative Memory Learning**
  - Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Cunxiao Du, Chao Du, Tianyu Pang, Zhuoran Yang, Mingyi Hong, Vincent Y. F. Tan (NUS, UMN, Sea AI Lab, Yale)
  - https://arxiv.org/abs/2509.26030
  - **Key insight**: Shows Muon enhances isotropy of weight matrices, leading to more balanced knowledge acquisition. Particularly effective at learning tail-end (rare/infrequent) associations in large language models compared to Adam, which tends to prioritize frequent patterns at the expense of rare ones.

- **How Muon's Spectral Design Benefits Generalization: A Study on Imbalanced Data**
  - Bhavya Vasudeva, Puneesh Deora, Yize Zhao, Vatsal Sharan, Christos Thrampoulidis (USC, UBC)
  - https://arxiv.org/abs/2510.22980
  - **Key insight**: Demonstrates that Muon/Shampoo learn all principal components at similar rates, unlike GD/Adam which prioritize dominant (majority-class) features first. On imbalanced datasets, Muon achieved 5%+ higher balanced accuracy by learning minority-class features earlier in training. This advantage is intrinsic to the orthogonalized update structure, not just learning rate tuning. Provides theoretical analysis with idealized Spectral Gradient Descent (SpecGD) showing the effect holds in linear and deep linear models.
 
- **Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data**
  - Chen Fan, Mark Schmidt, Christos Thrampoulidis (UBC)
  - https://arxiv.org/abs/2502.04664
  - **Key insight**: Analyzes the implicit bias of spectral descent methods (including Muon) on separable multiclass data. Characterizes what type of solutions these methods converge to compared to standard gradient descent.

## Critical Batch Size

Understanding optimal batch size scaling for Muon in large-scale training.

- **Optimal Scaling Needs Optimal Norm**
  - Oleg Filatov, Jiangtao Wang, Jan Ebert, Stefan Kesselheim (Julich Supercomputing Centre)
  - https://arxiv.org/abs/2510.03871
  - **Key insight**: Investigates the relationship between optimizer norms and optimal scaling behavior. Shows that different norms (including spectral norm used by Muon) affect the critical batch size and scaling efficiency differently.

- **Convergence Bound and Critical Batch Size of Muon Optimizer**
  - Naoki Sato, Hiroki Naganuma, Hideaki Iiduka (Meiji, Mila, Université de Montréal)
  - https://arxiv.org/abs/2507.01598
  - **Key insight**: Provides theoretical convergence proofs for Muon across multiple settings (including Nesterov momentum and weight decay variants). Derives the critical batch size that minimizes computational cost, offering practical guidance on hyperparameter interactions for efficient large-scale training.

## Empirical Evaluation

Comprehensive benchmarks and empirical studies comparing Muon to other optimizers.

- **Practical Efficiency of Muon for Pretraining**
  - Essential AI
  - https://arxiv.org/abs/2505.02222
  - **Key insight**: Demonstrates Muon's ability to expand the compute-time tradeoff frontier over AdamW by maintaining data efficiency at large batch sizes. Introduces telescoping algorithm for efficient hyperparameter transfer. Validates findings through extensive experiments on models up to 4B parameters, showing Muon maintains performance while reaching target loss faster.

- **Muon is Scalable for LLM Training**
  - Moonshot AI (Kimi2), UCLA
  - https://arxiv.org/abs/2502.16982
  - **Key insight**: Demonstrates successful scaling of Muon to 3B and 16B parameter MoE models. Introduces Moonlight implementation with (1) decoupled weight decay (crucial for implicit regularization) and (2) layer-wise update scaling. Achieves ~2X computational efficiency over AdamW (same loss with half the FLOPs) with better perplexity-compute Pareto frontier. Required minimal hyperparameter tuning, demonstrating robustness at scale.
 
- **Fantastic Pretraining Optimizers and Where to Find Them**
  - Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang (Stanford)
  - https://arxiv.org/abs/2509.02046
  - **Key insight**: Systematic benchmark of pretraining optimizers including Muon. Provides comparative analysis of convergence speed, final performance, and hyperparameter sensitivity across various model architectures and scales.

- **Benchmarking Optimizers for Large Language Model Pretraining**
  - Andrei Semenov, Matteo Pagliardini, Martin Jaggi (EPFL)
  - https://arxiv.org/abs/2509.01440
  - **Key insight**: Comprehensive benchmark showing that AdamW, AdaFactor, Lion perform similarly when well-tuned on transformer models up to 1.3B parameters. Finds optimizer choice affects solution nature more than raw pre-training loss, with potential downstream task performance differences.
 
- **Optimization Benchmark for Diffusion Models on Dynamical Systems**
  - Fabian Schaipp (Inria)
  - https://arxiv.org/abs/2510.19376v1
  - **Key insight**: Evaluates optimizers including Muon on diffusion model training for dynamical systems, extending empirical understanding beyond language/vision domains.

- **The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton**
  - Natalie Abreu, Nikhil Vyas, Sham Kakade, Depen Morwani (Harvard)
  - https://arxiv.org/abs/2510.09378
  - **Key insight**: Investigates second-order methods for LLMs, providing context for understanding Muon's position between first-order (Adam) and full second-order (Gauss-Newton) approaches.

- **What Really Matters in Matrix-Whitening Optimizers?**
  - Kevin Frans, Pieter Abbeel, Sergey Levine (UC Berkeley)
  - https://arxiv.org/abs/2510.25000
  - **Key insight**: Analyzes which components of matrix-whitening optimizers (like Shampoo and Muon) contribute most to performance. Helps identify the essential design principles that make spectrum-aware optimizers effective.

## Efficient Algorithm

Practical improvements reducing Muon's computational and memory costs.

- **LiMuon: Light and Fast Muon Optimizer for Large Models**
  - Feihu Huang, Yuning Luo, Songcan Chen (Nanjing U of Aeronautics and Astronautics)
  - https://arxiv.org/abs/2509.14562
  - **Key insight**: Introduces momentum-based variance reduction technique to Muon, enhancing convergence rates and reducing computational overhead. Discusses tradeoffs between full SVD and randomized SVD (RSVD) approaches, offering memory-efficient variants for large-scale models.

- **Effective Quantization of Muon Optimizer States**
  - Aman Gupta, Rafael Celente, Abhishek Shivanna, D. T. Braithwaite, Gregory Dexter, Shao Tang, Hiroto Udagawa, Daniel Silva, Rohan Ramanath, S. Sathiya Keerthi (Mubank, LinkedIn)
  - https://arxiv.org/abs/2509.23106
  - **Key insight**: Demonstrates that Muon's optimizer states can be effectively quantized without significant performance loss. Reduces memory footprint, making Muon more practical for memory-constrained environments and enabling larger models to be trained.

- **NorMuon: Making Muon more efficient and scalable**
  - Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao (Georgia Tech, Microsoft AI)
  - https://arxiv.org/abs/2510.05491
  - **Key insight**: Introduces neuron-wise normalization to address imbalances in neuron norms post-orthogonalization. Prevents certain neurons from dominating optimization while preserving Muon's conditioning benefits. Improves training efficiency and scalability, outperforming both Adam and original Muon in large-scale language model training.

## Distributed Setting

Adapting Muon for distributed and federated training scenarios.

- **Dion: Distributed Orthonormalized Updates**
  - Kwangjun Ahn, Byron Xu, Natalie Abreu, Ying Fan, Gagik Magakyan, Pratyusha Sharma, Zheng Zhan, John Langford (Microsoft Research AI Frontiers, Harvard)
  - https://arxiv.org/abs/2504.05295
  - **Key insight**: Extends orthogonalization techniques to distributed training settings. Addresses challenges of performing matrix operations across multiple workers while maintaining communication efficiency and convergence properties.
 
- **MuLoCo: Muon is a practical inner optimizer for DiLoCo**
  - Benjamin Thérien, Xiaolong Huang, Irina Rish, Eugene Belilovsky (Mila, UdM, Concordia)
  - https://arxiv.org/abs/2505.23725
  - **Key insight**: Shows that Muon works effectively as the inner optimizer in DiLoCo (Distributed Low-Communication) training framework. Demonstrates practical benefits for federated learning scenarios where communication between nodes is costly.

- **MuonBP: Faster Muon via Block-Periodic Orthogonalization**
  - Ahmed Khaled, Kaan Ozkara, Tao Yu, Mingyi Hong, Youngsuk Park (Princeton, AWS, UMN)
  - https://arxiv.org/abs/2510.16981
  - **Key insight**: Proposes block-periodic orthogonalization strategy that reduces frequency of expensive orthogonalization operations. Maintains Muon's benefits while significantly reducing computational overhead, especially beneficial in distributed settings.

## Scaling

Strategies for scaling second-order and matrix-based optimizers to very large models.

- **How to Scale Second-Order Optimization**
  - Zixi (Charlie) Chen, Shikai Qiu, Hoang Phan, Qi Lei, Andrew Gordon Wilson (NYU)
  - https://openreview.net/pdf/d9ff9b9df54dd1e155b0d792f9a86d879a81a53c.pdf
  - **Key insight**: Examines practical strategies for scaling second-order optimization methods (like those underlying Muon) to modern large-scale neural networks. Provides insights on memory management, computational tradeoffs, and approximation techniques.

## Regularization

Regularization techniques compatible with and enhancing Muon's implicit bias.

- **Cautious Weight Decay**
  - Lizhang Chen, Jonathan Li, Kaizhao Liang, Baiyu Su, Cong Xie, Nuo Wang Pierse, Chen Liang, Ni Lao, Qiang Liu (UT Austin, Google)
  - https://arxiv.org/abs/2510.12402
  - **Key insight**: Proposes Cautious Weight Decay, a refined weight decay strategy that adapts based on optimization dynamics. Particularly relevant for spectrum-aware optimizers like Muon, where decoupled weight decay is already important. Can further improve generalization by avoiding over-regularization of beneficial parameter directions.

## Enhancement

Extensions and complementary techniques that can be combined with Muon.

- **MARS: Unleashing the Power of Variance Reduction for Training Large Models**
  - Huizhuo Yuan, Yifeng Liu, Shuang Wu, Xun Zhou, Quanquan Gu (ByteDance Research, UCLA)
  - https://arxiv.org/abs/2411.10438
  - **Key insight**: Proposes MARS optimizer combining variance reduction with adaptive learning rates. Can potentially be integrated with Muon's spectral techniques for further improvements in large model training efficiency.

- **Training Deep Learning Models with Norm-Constrained LMOs**
  - Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, Volkan Cevher (EPFL, U Paris-Saclay)
  - https://arxiv.org/abs/2502.07529
  - **Key insight**: Proposes SCION using norm-constrained linear minimization oracles (LMOs). Shares conceptual ground with Muon's spectral norm constraint approach, offering alternative perspective on constrained optimization for neural networks.

- **REG: A Regularization Optimizer for Robust Training Dynamics**
  - Zehua Liu, Han Wu, Xiaojin Fu, Shuqi Liu, Xiongwei Han, Tao Zhong, Mingxuan Yuan (Huawei Noah's Ark Lab)
  - https://arxiv.org/abs/2510.03691
  - **Key insight**: Proposes REG (Regularized Gradient Descent) for more robust training dynamics. Explores explicit regularization schemes that could complement Muon's implicit spectral regularization.

- **Noise-Adaptive Layerwise Learning Rates: Accelerating Geometry-Aware Optimization**
  - Jie Hao, Xiaochuan Gong, Jie Xu, Zhengdao Wang, Mingrui Liu (George Mason)
  - https://arxiv.org/abs/2510.14009
  - **Key insight**: Proposes LANTON with noise-adaptive layer-wise learning rates. Compatible with geometry-aware optimizers like Muon, potentially offering additional per-layer adaptation on top of Muon's spectral conditioning.

## Blog Post

Accessible explanations and practical insights from practitioners.

- **Deep Learning Optimizers as Steepest Descent in Normed Spaces**
  - Franz Louis Cesista
  - https://leloykun.github.io/ponder/steepest-descent-opt/
  - **Key insight**: Educational blog explaining how modern optimizers (including Muon) can be viewed as steepest descent under different norm constraints. Makes the theoretical framework accessible to practitioners.
 
- **Muon and a Selective Survey on Steepest Descent in Riemannian and Non-Riemannian Manifolds**
  - Franz Louis Cesista
  - https://leloykun.github.io/ponder/steepest-descent-non-riemannian/
  - **Key insight**: Comprehensive blog surveying the geometric perspective on Muon and related optimizers. Connects Riemannian and non-Riemannian manifold optimization to practical deep learning.

- **Squeezing 1-2% Efficiency Gains Out of Muon by Optimizing the Newton-Schulz Coefficients**
  - Franz Louis Cesista
  - https://leloykun.github.io/ponder/muon-opt-coeffs/
  - **Key insight**: Practical investigation into optimizing the coefficients used in Muon's Newton-Schulz iteration. Shows that careful tuning can extract additional 1-2% efficiency gains, demonstrating room for implementation-level improvements.
  
## Related Work and Perspectives

Papers providing broader context, alternative approaches, or complementary insights.

- **Muon Optimizer Accelerates Grokking**
  - Amund Tveit, Bjørn Remseth, Arve Skogvold (Microsoft Norway)
  - https://arxiv.org/abs/2504.16041
  - **Key insight**: Investigates Muon's impact on grokking (delayed generalization phenomenon). Finds Muon significantly accelerates the transition from memorization to generalization compared to AdamW, reducing mean grokking epochs substantially across various configurations. Suggests spectral norm constraints and better conditioning facilitate faster onset of generalization.

- **Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization**
  - Dmitry Kovalev (Yandex Research)
  - https://arxiv.org/abs/2503.12645
  - **Key insight**: Provides trust-region optimization perspective on gradient orthogonalization techniques. Helps explain why orthogonalizing gradients (as Muon does) improves optimization through non-Euclidean geometry lens.

- **PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective**
  - Tim Tsz-Kit Lau, Qi Long, Weijie Su (U Penn)
  - https://arxiv.org/abs/2505.21799
  - **Key insight**: Proposes PolarGrad family of optimizers using a unifying preconditioning perspective. Places Muon within broader context of matrix-gradient methods, showing common theoretical foundations across different approaches.

- **The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm**
  - Noah Amsel, David Persson, Christopher Musco, Robert M. Gower (NYU, Flatiron Institute)
  - https://arxiv.org/abs/2505.16932
  - **Key insight**: Analyzes optimal methods for computing matrix sign function (core operation in Muon's orthogonalization). Provides theoretical analysis and practical improvements to Newton-Schulz iteration used in Muon, potentially enabling more efficient implementations.

- **Towards understanding of orthogonalization in Muon**
  - Valentyn Boreiko, Zhiqi Bu, Sheng Zha (U Tübingen, Amazon)
  - https://openreview.net/forum?id=4vzhqq5hpX (ICML 2025 Workshop)
  - **Key insight**: Workshop paper investigating the mechanisms by which orthogonalization in Muon affects training dynamics and final model properties. Contributes to theoretical understanding of why the technique works.
