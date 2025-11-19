# Patch Analysis & Explainability Framework (PaPr, GradCAM, Naive Bayes & Hybrid Model)

This project implements and compares multiple patch-level explainability and inference-reduction techniques, including **PaPr**, **Grad-CAM**, **Naive Bayes**, and a custom **Hybrid method** combining PaPr and Grad-CAM. The goal is to identify discriminative image regions, prune redundant patches, and evaluate how each method impacts model interpretability, speed, and accuracy.

-----

##  Features & Methods Overview

| Method | Type | Core Functionality |
| :--- | :--- | :--- |
| **PaPr (Ours)** | Training-free Pruning (CNN Feature-based) | Uses lightweight ConvNet features to derive **semantic importance** for efficient patch removal. |
| **Grad-CAM** | Gradient-based Attention (DL Interpretabilty) | Computes gradients w.r.t. specific activations to highlight **class-specific informative regions**. |
| **Naive Saturation** | Patch-level Baseline (Statistical) | A simple baseline that keeps areas based on **color intensity** for performance floor comparison. |
| **Hybrid (PaPr + Grad-CAM)** | Pruning Ensemble (Novel Contribution) | Multiplies PaPr's general object map with Grad-CAM’s fine-grained localization to create a **superior, high-accuracy mask**. |


-----

##  How to Run & Verify

### 1\. Quantitative Benchmark (The Thesis Proof)

To generate the final results table (including the Hybrid method) and save the CSV:

1.  Open the **`PaPr_Fianl.ipynb`** file.
2.  Run **all cells** sequentially (Cells 1 through 6).
      * **Result:** Generates the final quantitative data (Accuracy vs. Time) based on **10,000 images**.

### 2\. Visual Demonstration (The Panel Demo)

To launch the interactive app showing the 4-way comparison:

1.  Ensure you have completed the `ngrok` setup (as discussed previously).
2.  Run the application file:



```bash
streamlit run app.py
```
