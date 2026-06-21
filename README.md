# 22FAERS‑NLI
<img width="1391" height="730" alt="image" src="https://github.com/user-attachments/assets/c0294c34-a945-4d97-ab3b-a1883acd1ac2" />

Demo Video

https://github.com/user-attachments/assets/db804df1-4496-41f2-8126-0bbab1711f3f

# 22FAERS‑NLI — Automated Literature Validation for Pharmacovigilance in Ultra‑Rare Neurological Diseases

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/ZAM/22FAERS-NLI)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20359434.svg)](https://doi.org/10.5281/zenodo.20359434)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**22FAERS‑NLI** is a fully automated, zero‑cost pipeline that validates drug‑adverse event (AE) signals from the **FDA Adverse Event Reporting System (FAERS)** against **PubMed literature** using **natural language inference (NLI)**.  
It is specifically designed for **ultra‑rare neurological diseases**, where traditional pharmacovigilance methods struggle due to sparse data.

---

##  What it does

1. **Retrieves PubMed evidence** using a multi‑tier cascade query (drug + AE → safety‑qualified → broad).
2. **Baseline check:** keyword co‑occurrence.
3. **NLI classification:** a fine‑tuned **PubMedBERT** model (F1‑present = 0.714) scores the abstract with multi‑hypothesis prompting.
4. **Verdict:** *Present* (literature evidence supports the drug‑AE association) or *Absent*.

A live demo is available on 🤗 Hugging Face Spaces:  
 **[22FAERS‑NLI Space](https://huggingface.co/spaces/ZAM/22FAERS-NLI)**

---

##  Key Results (58 evidence‑available pairs, original gold standard)

| System | Accuracy | F1‑present | F1‑absent |
|--------|----------|------------|-----------|
| **Keyword co‑occurrence** | 0.667 | **0.716** | 0.596 |
| DeBERTa‑v3 (zero‑shot NLI) | 0.552 | 0.536 | 0.567 |
| PubMedBERT (zero‑shot NLI) | 0.526 | 0.571 | 0.471 |
| Groq Llama‑3.3‑70B (few‑shot) | 0.456 | 0.162 | 0.597 |
| **PubMedBERT fine‑tuned (5‑fold CV)** | **0.674 ± 0.196** | **0.714 ± 0.174** | **0.616 ± 0.230** |

> **Main finding:** Simple keyword matching outperforms a 70 B LLM (recall = 0.06) for signal confirmation.  
> Fine‑tuning a small biomedical NLI model (PubMedBERT) closes the gap with keyword and provides a practical, free solution.

---

##  Dataset

The gold standard was expanded from **85 manually curated pairs** (6 diseases) to **237 pairs across 22 ultra‑rare neurological diseases** via semi‑automated labelling and expert correction.

| Step | Pairs |
|------|-------|
| Original manual curation (6 diseases) | 85 |
| Phase 1 expansion (8 new diseases) | +104 |
| Phase 2 expansion (8 more diseases) | +43 |
| Deduplication / corrections | −7 |
| **Final gold standard** | **237** |
| Pairs with ≥1 PubMed abstract | 150 (68.4%) |

Diseases covered: Batten (CLN2), Sanfilippo (MPS III), Niemann‑Pick C, GM2 Gangliosidosis, Friedreich’s Ataxia, NBIA/PKAN, Metachromatic Leukodystrophy, Krabbe, MPS I, PKU Neurodegeneration, SMA, Fabry, Wilson’s, MPS IVA, Gaucher, Pompe, MPS II, MPS VI, Lysosomal Acid Lipase Deficiency, Hereditary Angioedema, Duchenne Muscular Dystrophy, Cystic Fibrosis.

---

##  Quick Start (local)

### 1. Clone the repository & install dependencies
    ```bash
      git clone https://github.com/your-username/22FAERS-NLI.git
      cd 22FAERS-NLI
      pip install -r requirements.txt


2. Download the fine‑tuned model
Place the pubmedbert_final/ folder (containing config.json, model.safetensors, tokenizer.json, vocab.txt, etc.) in the project root.
Alternatively, the app will automatically fall back to the zero‑shot PubMedBERT model if the fine‑tuned model is not found.

3. Launch the Gradio app
    ```bash
    python app.py
    Open http://127.0.0.1:7860 in your browser.

4. Test with built‑in examples
Click any example in the “Try these examples” panel and hit Validate signal. The fine‑tuned model will return a verdict with a confidence score.

## Repository Structure
    ```bash
    
    .
    ├── app.py                     # Gradio application (literature validator + dashboard)
    ├── requirements.txt           # Python dependencies
    ├── pubmedbert_final/          # Fine‑tuned PubMedBERT model (optional – will use zero‑shot if missing)
    ├── data/
    │   ├── full_dataset.csv       # 237‑pair gold standard
    │   └── checkpoint_*.pkl       # Serialised data for training/evaluation
    ├── figures/                   # All publication‑ready figures (PNG)
    ├── README.md
    └── LICENSE

##  Citation
If you use this work, please cite:

    ```bash
    
    @dataset{mansoori2026ntdpharmalit,
      author       = {Mansoori, Zaeem Ahmad},
      title        = {Automated Literature Validation for Pharmacovigilance Signals in Ultra‑Rare Neurological Diseases: A 4‑System Benchmark},
      year         = 2026,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.20359434},
      url          = {https://doi.org/10.5281/zenodo.20359434}
    }
    
The companion FAERS disproportionality analysis is archived at:
10.5281/zenodo.20353879

##  Author
> Zaeem Ahmad Mansoori
> Delhi Technological University, Delhi, India
> Pharmacovigilance · Natural Language Inference · Rare Disease Safety
