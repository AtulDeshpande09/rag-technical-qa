# Retrieval-Augmented Technical QA (RAG vs Fine-Tuning)

![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)
![Blog](https://img.shields.io/badge/Blog-Project%20Writeup-orange)

**Model:** https://huggingface.co/YOUR_HF_LINK
**Dataset:** https://www.kaggle.com/YOUR_KAGGLE_LINK
**Blog:** https://YOUR_BLOG_LINK

---

## Project Overview

This project investigates the effectiveness of **Retrieval-Augmented Generation (RAG)** and **parameter-efficient fine-tuning (LoRA)** for technical question answering across core Computer Science domains.

We evaluate four configurations of a large language model:

1. **Vanilla Mistral**
2. **RAG + Vanilla**
3. **LoRA Fine-Tuned**
4. **RAG + Fine-Tuned**

Our experiments reveal an important observation:

> Retrieval improves both lexical and semantic quality, while aggressive fine-tuning can degrade semantic coherence due to catastrophic forgetting.

---

## Dataset

The dataset contains **technical interview question–answer pairs** across core CS domains including:

* Data Structures & Algorithms
* Operating Systems
* Databases
* Computer Networks
* System Design

### Dataset Creation

The dataset was constructed in two stages:

1. **Seed Dataset**

   * ~200 curated Q&A pairs collected from technical interview resources.

2. **Synthetic Expansion**

   * Additional samples generated using **Qwen** to increase domain coverage and diversity.

---

### Data Preprocessing

A two-stage filtering pipeline was applied to ensure dataset quality.

**Exact Duplicate Removal**

| Stage                    | Samples |
| ------------------------ | ------- |
| Original dataset         | 2070    |
| Exact duplicates removed | 51      |
| Remaining samples        | 2019    |

**Semantic Filtering**

Semantic similarity filtering was performed using **MiniLM embeddings**.

| Stage                       | Samples  |
| --------------------------- | -------- |
| Input dataset               | 2019     |
| Semantic duplicates removed | 213      |
| Final dataset               | **1806** |

Similarity threshold used:

```
cosine similarity > 0.9
```

---

### Dataset Split

The final dataset was split using a fixed seed for reproducibility.

| Split      | Samples  |
| ---------- | -------- |
| Train      | 1264     |
| Validation | 270      |
| Test       | 272      |
| **Total**  | **1806** |

Dataset available here:

**Kaggle Dataset**

https://www.kaggle.com/YOUR_KAGGLE_LINK

---

## Model

Base model:

**Mistral-7B-Instruct**

Fine-tuning approach:

**LoRA (Low-Rank Adaptation)**

Retrieval system:

* Sentence Transformers (`all-MiniLM-L6-v2`)
* FAISS vector database
* Top-K semantic retrieval

Model available here:

**HuggingFace**

https://huggingface.co/YOUR_HF_LINK

---

## System Architecture

Pipeline:

User Question
↓
Semantic Retrieval (FAISS)
↓
Context Injection
↓
LLM Generation
↓
Final Answer

---

## Training Curve

![Training Loss](visualization/loss_curve.png)

The LoRA model converges rapidly within a few epochs due to the small domain-specific dataset.

---

## Quantitative Results

| Model             | BLEU-4     | ROUGE-L   | BERTScore F1 | Exact Match |
| ----------------- | ---------- | --------- | ------------ | ----------- |
| Vanilla           | 0.0274     | 0.213     | 0.929        | 0.000       |
| **RAG + Vanilla** | **0.0515** | **0.298** | **0.890**    | **0.007**   |
| Fine-Tuned        | **0.0561** | 0.287     | 0.889        | 0.007       |
| RAG + Fine-Tuned  | 0.0380     | 0.252     | 0.871        | 0.003       |

Metric visualization:

![Metric Comparison](visualization/metrics_comparison.png)

### Key Observations

* **RAG improves both lexical and semantic similarity metrics**
* Fine-tuning improves lexical overlap but harms semantic coherence
* Combining RAG with a fine-tuned model does **not produce synergistic benefits**

---

## Example Inference

These examples demonstrate the qualitative differences between configurations.

---

### Example 1

**Question**

Implement a function to check if a binary tree is balanced.

**RAG + Fine-Tuned**

```
A Binary Search Tree is a binary tree where each node has a key and the keys are sorted in ascending order.

A Hash Table is a data structure that uses a hash function to map keys to values.
```

**RAG + Vanilla**

```
To check if a binary tree is balanced, you can use the following recursive function:

public boolean isBalanced(Node node) {
    if (node == null) return true;

    int leftHeight = height(node.left);
    int rightHeight = height(node.right);

    if (Math.abs(leftHeight - rightHeight) > 1)
        return false;
}
```

Reference

```
Use a recursive function to check the height of each subtree; return false if the difference is more than one.
```

---

### Example 2

**Question**

Explain pathway analysis.

**RAG + Fine-Tuned**

```
GO analysis investig
```

**RAG + Vanilla**

```
Pathway analysis interprets biological meaning of molecular data by mapping it to known biochemical pathways using databases such as KEGG or Reactome.
```

---

### Example 3

**Question**

When would you choose JWT over Session Cookies?

**RAG + Fine-Tuned**

```
JWT is a compact,
```

**RAG + Vanilla**

```
JWTs are preferred when stateless APIs are used and server-side sessions are undesirable, especially in microservices architectures.
```

---

### Example 4

**Question**

Describe seasonality in data.

**RAG + Fine-Tuned**

*(empty output)*

**RAG + Vanilla**

```
Seasonality refers to recurring patterns in time series data that occur at regular intervals such as monthly or yearly.
```

---

### Example 5

**Question**

Design a real-time multiplayer online game architecture.

**RAG + Fine-Tuned**

```
Consider a hybrid architecture combining client-
```

**RAG + Vanilla**

```
Use a client-server architecture with real-time state synchronization, load balancing, and horizontal scaling to support large numbers of concurrent players.
```

---

## Key Insight

Our experiments reveal a **fine-tuning paradox**:

* Fine-tuning improves lexical similarity metrics
* However, semantic coherence degrades
* Retrieval grounding significantly improves answer quality

Therefore:

**RAG + Vanilla LLM provides the most reliable configuration for technical question answering.**

---

## License

Released for **research and educational purposes**.
