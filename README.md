
**gpt2-moe-detector**

---

# GPT‑2 Text Detection via Heterogeneous Mixture‑of‑Experts

A Python implementation of a four‑expert Mixture‑of‑Experts (MoE) model to distinguish between human‑written text and GPT‑2‑generated text. Combines lexical, semantic, stylometric, and zero‑shot perplexity cues with a learned gating network for robust, interpretable detection.

---

## 🚀 Features

- **Lexical Expert**  
  TF‑IDF vectorization + logistic regression for n‑gram anomaly detection.

- **Semantic Expert**  
  Fine‑tuned DistilBERT classifier for deep contextual understanding.

- **Stylometric Expert**  
  Random forest on structural/style features (sentence length, TTR, entropy, etc.).

- **Zero‑Shot Expert**  
  Maps GPT‑2 perplexity via a logistic function—no additional training required.

- **Gating Network**  
  Soft‑EM trained MLP with learnable temperature scaling to route each sample to the most confident expert.

- **Interpretable Routing**  
  Per‑sample expert assignments and confidence scores enable dynamic human‑in‑the‑loop review.

---

## 🛠️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/CooperJRG/gpt2-moe-detector.git
   cd gpt2-moe-detector
   ```

2. **Create a virtual environment & install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Prepare data**

   * Place `small-117M.train.jsonl` and `webtext.train.jsonl` under `data/`.
   * (Optionally) adjust sampling sizes in `notebooks/PML_project.ipynb`.

---

## ▶️ Quickstart

Launch the Colab notebook for an end‑to‑end walkthrough:

```bash
jupyter notebook notebooks/PML_project.ipynb
```

Key steps include:

1. **Data Loading & Sampling**
2. **Feature Extraction**
3. **Expert Training (LogReg, RF, DistilBERT)**
4. **Zero‑Shot Perplexity Scoring**
5. **Gating Network & EM Training**
6. **Final Evaluation**

Results (on held‑out test set):

```
Full‑scale (160 K/20 K/20 K): Acc = 0.9058, F1 = 0.9024  
Mini‑run (2 K samples):      Acc = 0.8935, F1 = 0.8898  
```

---

## 📊 Results & Analysis

* **Expert Utilization**

  * DistilBERT & Stylometric RF dominate, TF‑IDF and zero‑shot lightly used.
* **Error‑Bucket Analysis**

  * Accuracy stratified by gating confidence bins (see appendix).
* **Interpretable Insights**

  * Per‑sample gating weights reveal when to defer to human review.

---

## 🔍 Architecture Overview

1. **Experts**

   * Each expert outputs `[P(human), P(AI)]`.
2. **Meta‑Features**

   * Entropy & margin for each expert → 8 dims.
   * Concatenate with DistilBERT embedding and stylometric vector.
3. **Gating MLP**

   * 2‑layer network + BatchNorm, Dropout, temperature‑scaled softmax.
4. **EM Loop**

   * E‑step: compute responsibilities.
   * M‑step: re‑train LogReg & RF with weights, fine‑tune DistilBERT, update gate.


## ✉️ Contact

Cooper Gilkey — [cooperjgilkey@gmail.com](mailto:cooperjgilkey@gmail.com)
GitHub: [github.com/CooperJRG](https://github.com/CooperJRG)
LinkedIn: [linkedin.com/in/cooper-gilkey](https://www.linkedin.com/in/cooper-gilkey)
