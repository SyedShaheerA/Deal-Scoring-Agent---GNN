# 🏢 Deal Scoring Agent — GNN

Ranks business listings by fit against a buyer's criteria using a **Graph Neural Network (GageSAGE)** to identify market "comparables" and hidden value.

---

## 🚀 Overview

This agent automates the deal screening process for private equity or individual searchers. Unlike traditional filters that use static rules, this model builds a relational graph of the market. It learns that a "good deal" isn't just about the numbers in isolation, but how those numbers compare to similar listings in the same industry and price bracket.

### Key Features
* **Synthetic Data Engine**: Generates 120+ realistic BizBuySell-style listings with metrics like EBITDA multiples, growth rates, and owner involvement.
* **Relational Graph Mapping**: Automatically links "comparable" deals based on industry and capital requirements.
* **GraphSAGE Architecture**: Uses a 2-layer Graph Neural Network to compress complex business data into a 32-dimensional "deal embedding".
* **Intelligent Ranking**: Combines hard-constraint filtering with GNN similarity scoring to find the 100% match for a buyer's specific profile.

---

## 🛠️ Technical Pipeline

1.  **Node Features**: Each business is represented by a vector of 25 features, including normalized financials and one-hot encoded industry/location data.
2.  **Edge Logic**: Edges are drawn between listings if they share an industry and have an asking price within 50% of each other, simulating a "market comp".
3.  **Self-Supervised Learning**: The GNN is trained to reconstruct node features from their neighbors, forcing the model to learn the structural "context" of a deal.
4.  **Inference**: A buyer profile is projected into the learned embedding space to calculate a `gnn_score` based on cosine similarity.

---

## 📦 Requirements

To run the notebook, you will need:
* `torch`
* `torch-geometric`
* `pandas`
* `numpy`
* `scikit-learn`

```bash
pip install torch torch-geometric pandas numpy scikit-learn
