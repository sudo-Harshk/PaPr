# PaPr: Critical Architectural Analysis & Optimization of Training-Free Pruning

This repository contains the code, quantitative benchmarks, and interactive visual audit for my M.Tech thesis project. The research evaluates the architectural vulnerabilities of training-free patch pruning (PaPr) and proposes dynamic optimization strategies to recover accuracy and stabilize inference across diverse visual domains.

---

## The 5 Research Objectives

This project moves beyond standard patch pruning by conducting a rigorous 5-stage critical analysis of the framework:

1. **Quantification of Saliency Drift (The Audit):** Measures the architectural disagreement (Intersection over Union) between the lightweight scoring model (MobileNetV2) and the backbone classifier (ResNet50).
2. **Mitigation of Budget Rigidity (Optimization):** Implements an **Entropy-Aware Dynamic Controller** that calculates the visual variance ($\sigma$) of a scene to autonomously adjust the patch pruning budget, protecting complex object boundaries.
3. **Saliency Fusion for Accuracy Recovery (Novel Strategy):** Introduces a **Hybrid Fusion Method** that element-wise multiplies general semantic features (PaPr) with decision-specific gradients (Grad-CAM), successfully mitigating Saliency Drift and recovering near-baseline accuracy.
4. **Heuristic Baseline Ablation (Robustness Proof):** Evaluates naive statistical baseline methods (e.g., color saturation/edge density) to mathematically prove that effective pruning requires deep-feature semantic awareness.
5. **Inference Latency Analysis (Practical Trade-off):** Quantifies the real-world computational bottleneck, proving that the overhead of generating saliency maps limits the practical speedup of training-free GFLOP reduction on standard GPU architectures.

---

## Methodology & Frameworks

| Method | Core Functionality | Thesis Contribution |
| :--- | :--- | :--- |
| **Baseline (ResNet50)** | Standard Forward Pass | Establishes the 1.0x Speedup / 0% Accuracy Drop benchmark. |
| **PaPr (Static)** | Semantic Feature Pruning | Analyzes the standard vulnerability of fixed "keep-ratios." |
| **PaPr (Dynamic)** | Entropy-Aware Budgeting | Solves Obj 2 via standard deviation ($\sigma$) adaptive logic. |
| **Hybrid (Fusion)** | PaPr + Grad-CAM Matrix | Solves Obj 3 by recovering accuracy lost to architectural drift. |
| **Naive Saturation** | Pixel-Level Heuristics | Solves Obj 4 by proving superficial pruning fails catastrophically. |

---

## How to Run & Verify

### 1. Quantitative Benchmark (The Hard Data)
To regenerate the final 5-Objective CSV report:
1. Open the **`PaPr_Final.ipynb`** notebook.
2. Run the environment setup and data loading cells.
3. Execute the `Master Benchmark Functions` and the `Comprehensive 5-Objective Research Audit` cells.
4. **Result:** Outputs the final thesis data table and saves `final_mtech_audit_results.csv`.

### 2. Interactive Architectural Audit (The Live Demo)
To launch the real-time visual workstation used for the Viva defense:
1. Ensure your local environment (or Kaggle instance) has `streamlit` and `pyngrok` installed.
2. Run the application via terminal:
```bash
streamlit run app.py