# Tiny Recursive Models (TRM) for SME Decision Support

**Author:** Santiago Leonardo Hernández Beltrán  
**Affiliation:** Faculty of ICT Engineering, Universidad Pontificia Bolivariana (UPB), Medellín, Colombia  

📄 **Full Paper:** [Read the full research paper (PDF)](./docs/Hernandez_TRM_SME_Paper.pdf)

---

## Overview

Small and medium-sized enterprises (SMEs) in Colombia account for 99.5% of formal businesses but often lack the historical data, GPU hardware, and stable internet connectivity required to deploy cloud-based AI models. 

Inspired by Jolicoeur-Martineau’s **Tiny Recursive Models (TRM)**, this project evaluates whether a lightweight, 2-layer recursive neural network can be trained and executed locally on a standard CPU for tabular business decision-making (credit risk detection) under strict data and compute constraints.

---

## Benchmark Results (CPU, Offline)

Evaluated on the **German Credit dataset** (`credit-g`, $N = 1,000$ samples, 61 processed features, 80/20 stratified split) using a standard CPU:

| Model | Accuracy | F1-Score | Training Time | Inference Latency | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.7350** | **0.8166** | **0.10 s** | 0.0165 ms/sample | — |
| **Standard MLP** | 0.6800 | 0.7681 | 2.22 s | **0.0029 ms/sample** | 8,193 |
| **Tabular TRM ($T=4$)** | 0.6750 | 0.7653 | 5.05 s | 0.0109 ms/sample | 16,641 |

**Key Takeaway:** Tabular TRM trains in ~5 seconds and achieves sub-millisecond offline CPU inference (0.0109 ms/sample), proving local deployment feasibility without cloud APIs or GPUs. However, gradient-boosted trees (XGBoost) remain more sample-efficient for small, static tabular datasets.

---

## Project Structure

```text
TRMProposal/
├── data/
│   ├── 01_raw/             # Raw datasets
│   └── 08_reporting/       # Saved metrics, tables, and plots
├── src/
│   └── trm_benchmark.py    # Benchmark script (XGBoost, MLP, Tabular TRM)
├── .gitignore
├── Makefile
├── README.md
└── requirements.txt
```

---

## How to Reproduce

### Prerequisites
* Python 3.10+ (with `python3-venv` installed)
* `make`

### Run the Benchmark

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SantiagoHB/TRMProposal.git](https://github.com/SantiagoHB/TRMProposal.git)
   cd TRMProposal
   ```

2. **Set up the virtual environment, install dependencies, and run the experiment:**
   ```bash
   make all
   ```

3. **To re-run only the benchmark after setup:**
   ```bash
   make run
   ```

4. **To clean up the virtual environment and cache files:**
   ```bash
   make clean
   ```