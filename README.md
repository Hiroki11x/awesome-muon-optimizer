# Awesome Muon Optimizer [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

(PR welcome!)

## Original Literature

- Muon: An optimizer for hidden layers in neural networks
  - Keller Jordan
  - OpenAI
  - https://kellerjordan.github.io/posts/muon/

- Deriving Muon
  - Jeremy Bernstein 
  - MIT
  - https://jeremybernste.in/writing/deriving-muon
 
- Modular Manifolds
  - Jeremy Bernstein 
  - Thinking Machine
  - https://thinkingmachines.ai/blog/modular-manifolds/

- Old Optimizer, New Norm: An Anthology
  - Jeremy Bernstein, Laker Newhouse
  - MIT
  - https://arxiv.org/abs/2409.20325
 
- Scalable Optimization in the Modular Norm
  - Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, Jeremy Bernstein
  - MIT
  - https://arxiv.org/abs/2405.14813

- Duality, Weight Decay, and Metrized Deep Learning
  - Laker Newhouse
  - MIT
  - https://www.lakernewhouse.com/thesis.pdf
 
- Understanding Muon Chapter 1: Into the Matrix
  - Laker Newhouse
  - MIT
  - https://www.lakernewhouse.com/writing/muon-1
 
- Depths of First-Order Optimization
  - Jeremy Bernstein
  - MIT
  - https://docs.google.com/presentation/d/1PIAChMGGwhmdUxDPyOo1o8Qlhq3h_ofV2mhBb6JHH04

## Theoretical Analysis

- On the Convergence Analysis of Muon
  - Da Chang, Yongxiang Liu, Ganzhao Yuan
  - University of Virginia, University of British Columbia, Meta, University of Wisconsin-Madison
  - https://arxiv.org/abs/2505.23737

- A Note on the Convergence of Muon
  - Jiaxiang Li, Mingyi Hong
  - University of Minnesota
  - https://arxiv.org/abs/2502.02900

- Muon Optimizes Under Spectral Norm Constraints
  - Lizhang Chen, Jonathan Li, Qiang Liu
  - UT Austin
  - https://arxiv.org/abs/2506.15054

## Understanding Property

- Muon Outperforms Adam in Tail-End Associative Memory Learning
  - Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Cunxiao Du, Chao Du, Tianyu Pang, Zhuoran Yang, Mingyi Hong, Vincent Y. F. Tan
  - National University of Singapore, University of Minnesota, Sea AI Lab, Yale University
  - https://arxiv.org/abs/2509.26030

## Critical Batch Size

- Optimal Scaling Needs Optimal Norm
  - Oleg Filatov, Jiangtao Wang, Jan Ebert, Stefan Kesselheim
  - Julich Supercomputing Centre
  - https://arxiv.org/abs/2510.03871

- Convergence Bound and Critical Batch Size of Muon Optimizer
  - Naoki Sato, Hiroki Naganuma, Hideaki Iiduka
  - UMeiji, Mila, Université de Montréal
  - https://arxiv.org/abs/2507.01598

## Empirical Evaluation

- Practical Efficiency of Muon for Pretraining
  - Essential AI
  - https://arxiv.org/abs/2505.02222

- Muon is Scalable for LLM Training
  - Moonshot AI (Kimi2), UCLA 
  - https://arxiv.org/abs/2502.16982
 
- Fantastic Pretraining Optimizers and Where to Find Them
  - Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang
  - Stanford
  - https://arxiv.org/abs/2509.02046

- Benchmarking Optimizers for Large Language Model Pretraining
  - Andrei Semenov, Matteo Pagliardini, Martin Jaggi
  - EPFL
  - https://arxiv.org/abs/2509.01440

## Efficient Algorithm

- LiMuon: Light and Fast Muon Optimizer for Large Models
  - Nanjing University of Aeronautics and Astronautics
  - Feihu Huang, Yuning Luo, Songcan Chen
  - https://arxiv.org/abs/2509.14562

- Effective Quantization of Muon Optimizer States
  - Mubank, LinkedIn
  - Aman Gupta, Rafael Celente, Abhishek Shivanna, D. T. Braithwaite, Gregory Dexter, Shao Tang, Hiroto Udagawa, Daniel Silva, Rohan Ramanath, S. Sathiya Keerthi
  - https://arxiv.org/abs/2509.23106

- NorMuon: Making Muon more efficient and scalable
  - Georgia Tech, Microsoft AI
  - Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao
  - https://arxiv.org/abs/2510.05491

## Distributed Setting

- Dion: Distributed Orthonormalized Updates
  - Kwangjun Ahn, Byron Xu, Natalie Abreu, Ying Fan, Gagik Magakyan, Pratyusha Sharma, Zheng Zhan, John Langford
  - Microsoft Research (AI Frontiers), Harvard University
  - https://arxiv.org/abs/2504.05295
 
- MuLoCo: Muon is a practical inner optimizer for DiLoCo
  - Benjamin Thérien, Xiaolong Huang, Irina Rish, Eugene Belilovsky
  - Mila, Université de Montréal, Concordia
  - https://arxiv.org/abs/2505.23725

## Enhancement

- MARS: Unleashing the Power of Variance Reduction for Training Large Models
  - Huizhuo Yuan, Yifeng Liu, Shuang Wu, Xun Zhou, Quanquan Gu
  - ByteDance Research, UCLA
  - https://arxiv.org/abs/2411.10438
  - Propose MARS

- Training Deep Learning Models with Norm-Constrained LMOs
  - Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, Volkan Cevher
  - EPFL, Universite Paris-Saclay.
  - https://arxiv.org/abs/2502.07529
  - Propose SCION

- REG: A Regularization Optimizer for Robust Training Dynamics
  - Zehua Liu, Han Wu, Xiaojin Fu, Shuqi Liu, Xiongwei Han, Tao Zhong, Mingxuan Yuan
  - Huawei Noah’s Ark Lab
  - https://arxiv.org/abs/2510.03691
  - Propose REG (Regularized gradient descent)

## Blog Post

- Deep Learning Optimizers as Steepest Descent in Normed Spaces
  - Franz Louis Cesista
  - https://leloykun.github.io/ponder/steepest-descent-opt/
 
- Muon and a Selective Survey on Steepest Descent in Riemannian and Non-Riemannian Manifolds
  - Franz Louis Cesista
  - https://leloykun.github.io/ponder/steepest-descent-non-riemannian/

- Squeezing 1-2% Efficiency Gains Out of Muon by Optimizing the Newton-Schulz Coefficients
  - Franz Louis Cesista
  - https://leloykun.github.io/ponder/muon-opt-coeffs/
  
## Unclassified


- Muon Optimizer Accelerates Grokking
  - Amund Tveit, Bjørn Remseth, Arve Skogvold
  - Microsoft (Norway)
  - https://arxiv.org/abs/2504.16041

- Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization
  - Dmitry Kovalev
  - Yandex Research
  - https://arxiv.org/abs/2503.12645

- PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective
  - Tim Tsz-Kit Lau, Qi Long, Weijie Su
  - University of Pennsylvania
  - https://arxiv.org/abs/2505.21799

- The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm
  - Noah Amsel, David Persson, Christopher Musco, Robert M. Gower
  - New York University, Flatiron Institute
  - https://arxiv.org/abs/2505.16932

- Towards understanding of orthogonalization in Muon
  - Valentyn Boreiko, Zhiqi Bu, Sheng Zha
  - University of Tübingen, Amazon
  - https://openreview.net/forum?id=4vzhqq5hpX (ICML2025 WS)
