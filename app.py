# ============================================================
# 22FAERS‑NLI — Automated Literature Validation for Pharmacovigilance
# ============================================================
import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Bio import Entrez
import time
import re
import os
import plotly.graph_objects as go

Entrez.email = "your.email@dtu.ac.in"   # ← replace

# ── Model selection ──────────────────────────────────────────
MODEL_PATH = "pubmedbert_final" if os.path.exists("pubmedbert_final/config.json") else "pritamdeka/PubMedBERT-MNLI-MedNLI"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

HYPOTHESES = [
    "{drug} causes {ae}",
    "{drug} is associated with {ae}",
    "{drug} induced {ae}",
    "{drug} was reported to cause {ae}",
]
NEG_HYP = "{drug} does not cause {ae}"
THRESHOLD = 0.25

if MODEL_PATH == "pubmedbert_final":
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("✅ Loaded fine‑tuned PubMedBERT (F1‑present = 0.714)")
    USE_FINETUNED = True
else:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("⚠️  Using zero‑shot PubMedBERT (fine‑tuned model not found)")
    USE_FINETUNED = False

# ── Best‑sentence extraction ──────────────────────────────────
def best_sentence(text, drug, ae):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    d, a = drug.lower(), ae.lower()
    matches = [s for s in sentences if d in s.lower() and a in s.lower()]
    if matches:
        return max(matches, key=len)[:800]
    return text[:800]

# ── PubMed retrieval ─────────────────────────────────────────
def fetch_abstracts(drug, ae, max_results=5):
    results = []
    queries = [
        f'"{drug}" AND "{ae}"',
        f'{drug} AND {ae} AND (adverse OR "side effect" OR safety)',
        f'{drug} AND {ae}',
    ]
    for q in queries:
        try:
            handle = Entrez.esearch(db="pubmed", term=q, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            ids = record["IdList"]
            if not ids: continue
            fetch = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
            raw = fetch.read()
            chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 60]
            for pmid, chunk in zip(ids, chunks[:max_results]):
                results.append({"pmid": pmid, "text": chunk[:2000]})
            if results: break
            time.sleep(0.4)
        except: continue
    return results

def keyword_match(drug, ae, text):
    return drug.lower() in text.lower() and ae.lower() in text.lower()

# ── Model inference ──────────────────────────────────────────
def score_text(drug, ae, text):
    if not text.strip():
        return "Absent (no evidence)", 0.0
    text = best_sentence(text, drug, ae)

    if USE_FINETUNED:
        scores = []
        for hyp in HYPOTHESES:
            inp = tokenizer(text, hyp.format(drug=drug, ae=ae), return_tensors="pt",
                            truncation=True, max_length=512, padding=True)
            with torch.no_grad(): logits = model(**inp).logits
            probs = torch.softmax(logits, -1).squeeze(0)
            scores.append(probs[1].item())
        neg_inp = tokenizer(text, NEG_HYP.format(drug=drug, ae=ae), return_tensors="pt",
                            truncation=True, max_length=512, padding=True)
        with torch.no_grad(): logits = model(**neg_inp).logits
        probs = torch.softmax(logits, -1).squeeze(0)
        neg_prob = probs[1].item()
        avg = sum(scores) / len(scores) if scores else 0.0
        raw_confidence = max(0, avg - 0.5 * neg_prob)
        label = "Present" if raw_confidence >= THRESHOLD else "Absent"
        display_conf = min(1.0, raw_confidence / 0.5)
        return label, display_conf
    else:
        scores = []
        for hyp in HYPOTHESES:
            inp = tokenizer(text, hyp.format(drug=drug, ae=ae), return_tensors="pt",
                            truncation=True, max_length=512, padding=True)
            with torch.no_grad(): logits = model(**inp).logits
            probs = torch.softmax(logits, -1).squeeze(0)
            for i, l in model.config.id2label.items():
                if l.lower() == "entailment": scores.append(probs[i].item())
        neg_inp = tokenizer(text, NEG_HYP.format(drug=drug, ae=ae), return_tensors="pt",
                            truncation=True, max_length=512, padding=True)
        with torch.no_grad(): logits = model(**neg_inp).logits
        probs = torch.softmax(logits, -1).squeeze(0)
        neg = sum(probs[i].item() for i, l in model.config.id2label.items() if l.lower() == "entailment")
        avg = sum(scores) / len(scores) if scores else 0.0
        raw_confidence = max(0, avg - 0.5 * neg)
        label = "Present" if raw_confidence >= THRESHOLD else "Absent"
        return label, raw_confidence

# ── Main validation function ──────────────────────────────────
def run_validation(drug, ae, abstract):
    drug, ae = drug.strip(), ae.strip()
    abstract = abstract.strip()
    if not drug or not ae:
        return "", "", "", "", ""

    if abstract:
        kw = "✅ Present" if keyword_match(drug, ae, abstract) else "❌ Absent"
        label, conf = score_text(drug, ae, abstract)
        abstract_html = ""
        tier_info = "Using manually entered abstract."
    else:
        abstracts = fetch_abstracts(drug, ae)
        if abstracts:
            best_label, best_conf = "Absent (no evidence)", 0.0
            best_idx = 0
            for i, a in enumerate(abstracts):
                lbl, c = score_text(drug, ae, a["text"])
                if c > best_conf:
                    best_conf = c
                    best_label = lbl
                    best_idx = i
            kw = "✅ Present" if keyword_match(drug, ae, abstracts[best_idx]["text"]) else "❌ Absent"
            label, conf = best_label, best_conf
            abstract_html = ""
            for a in abstracts[:3]:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/"
                t = re.sub(rf'(?i)({re.escape(drug)})', r'<mark>\1</mark>', a["text"])
                t = re.sub(rf'(?i)({re.escape(ae)})', r'<mark style="background:#ffd6d6">\1</mark>', t)
                abstract_html += f'<div class="abstract-card"><a class="pmid-link" href="{url}" target="_blank">PMID: {a["pmid"]} ↗</a><p class="abstract-text">{t}…</p></div>'
            tier_info = f"Retrieved {len(abstracts)} abstracts, best score: {best_conf:.1%}"
        else:
            kw = "❌ Absent"
            label, conf = "Absent (no evidence)", 0.0
            abstract_html = '<p class="no-result">No PubMed abstracts found for this pair.</p>'
            tier_info = "Retrieved 0 abstracts."

    model_str = f"{'✅' if label == 'Present' else '❌'} {label}  (confidence: {conf:.1%})"
    stamp_html = f'<div class="stamp {"present" if label=="Present" else "absent"}">{label.upper()}</div>'
    return kw, model_str, abstract_html, tier_info, stamp_html

# ── MODERN, CLEAN CSS ──────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600;700&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body, .gradio-container {
  width: 100vw !important; max-width: 100% !important; min-height: 100vh;
  margin: 0 !important; padding: 0 !important;
  background: #f8f7f4 !important;
  font-family: 'Inter', sans-serif !important;
  color: #1C3F60 !important;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}
.gradio-container { padding-top: 70px !important; }

.navbar {
  position: fixed; top: 0; left: 0; width: 100%; z-index: 1000;
  background: rgba(255,255,255,0.95); backdrop-filter: blur(10px);
  border-bottom: 1px solid #e5e0d8; padding: 14px 5%;
  display: flex; justify-content: space-between; align-items: center;
}
.nav-logo { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700; color: #1C3F60; }
.nav-logo span { color: #C44536; }
.nav-links { display: flex; gap: 2rem; list-style: none; }
.nav-links a { text-decoration: none; color: #1C3F60; font-weight: 600; font-size: 0.9rem; padding: 6px 0; border-bottom: 2px solid transparent; transition: border-color 0.2s, color 0.2s; }
.nav-links a:hover { border-color: #C44536; color: #C44536; }

.section { padding: 60px 5% 40px; width: 100%; max-width: 100%; margin: 0; }
.section-title {
  font-family: 'Playfair Display', serif; font-size: 2.5rem; color: #1C3F60;
  border-bottom: 2px solid #D4A843; padding-bottom: 10px; margin-bottom: 30px; width: 100%;
}
.hero { text-align: center; }
.hero h1 { font-size: 3.5rem; margin-bottom: 10px; }
.hero p { color: #1C3F60; max-width: 700px; margin: 0 auto 30px; font-size: 1.1rem; }

.paper-card {
  background: #ffffff; border: 1px solid #e5e0d8; border-radius: 16px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.04); padding: 30px 5%; margin: 0 auto 24px;
  width: 100%; max-width: 100%; box-sizing: border-box;
}

.gradio-container input[type="text"], .gradio-container textarea {
  background: #f9f9f9 !important; border: 1px solid #d3cfc7 !important;
  border-radius: 10px !important; color: #1C3F60 !important; font-size: 1rem !important;
  padding: 14px 18px !important; font-family: 'Inter', sans-serif !important;
  width: 100% !important; box-shadow: none !important; transition: border-color 0.2s;
}
.gradio-container input[type="text"]:focus, .gradio-container textarea:focus {
  border-color: #1C3F60 !important; outline: none !important;
}
.gradio-container label {
  color: #1C3F60 !important; font-size: 0.85rem !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 0.6px !important; margin-bottom: 8px !important;
}
.gradio-container .output-textbox, .gradio-container .gr-text-input {
  background: #f9f9f9 !important; border: 1px solid #d3cfc7 !important;
  border-radius: 10px !important; color: #1C3F60 !important; font-size: 1rem !important; width: 100% !important;
}

/* ── VIBRANT GRADIENT BUTTON ── */
button[id*="run"], .gradio-button.primary {
  width: 100% !important;
  background: linear-gradient(135deg, #1C3F60 0%, #2A5A8C 100%) !important;
  border: none !important;
  border-radius: 50px !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 1.1rem !important;
  padding: 16px 32px !important;
  cursor: pointer !important;
  letter-spacing: 0.4px !important;
  box-shadow: 0 4px 15px rgba(28,63,96,0.3) !important;
  transition: all 0.3s ease;
  text-transform: none !important;
}
button[id*="run"]:hover, .gradio-button.primary:hover {
  background: linear-gradient(135deg, #152C42 0%, #1F4A6B 100%) !important;
  box-shadow: 0 8px 25px rgba(28,63,96,0.4) !important;
  transform: translateY(-2px);
}
button[id*="run"]:active { transform: scale(0.98) !important; }

/* ── EXAMPLES PANEL ── */
.gr-examples {
  background: #f0ede6 !important;
  border: 1px solid #d3cfc7 !important;
  border-radius: 12px !important;
  padding: 16px !important;
  margin-top: 16px !important;
}
.gr-examples label {
  color: #1C3F60 !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  opacity: 1 !important;
}

/* ── STAMP ── */
.stamp {
  display: inline-block; font-family: 'Playfair Display', serif; font-size: 1.8rem;
  font-weight: 700; letter-spacing: 2px; color: #C44536;
  border: 3px solid #C44536; border-radius: 50%; padding: 14px 24px;
  transform: rotate(-8deg); opacity: 0.9;
  box-shadow: 0 2px 8px rgba(196,69,54,0.3); margin: 20px 0 0; text-align: center;
}
.stamp.absent { color: #4A4A4A; border-color: #4A4A4A; box-shadow: 0 2px 8px rgba(74,74,74,0.2); }

/* ── ABSTRACT CARDS ── */
.abstract-card {
  background: #fdfcf9; border-left: 4px solid #1C3F60; border-radius: 10px;
  padding: 16px 20px; margin: 12px 0; font-size: 0.95rem; width: 100%;
}
.pmid-link { font-size: 0.85rem; font-weight: 600; color: #1C3F60; text-decoration: none; }
.pmid-link:hover { text-decoration: underline; }
.abstract-text { color: #1C3F60; line-height: 1.7; margin-top: 8px; }
.abstract-text mark { background: #E8E0D5; color: #1C3F60; border-radius: 2px; padding: 0 3px; }
.no-result { color: #4A4A4A; font-style: italic; padding: 12px 0; }

/* ── STEPS GRID ── */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 24px; width: 100%; }
.step-card {
  background: #ffffff; border: 1px solid #e5e0d8; border-radius: 12px;
  padding: 30px; text-align: center; transition: transform 0.3s ease;
}
.step-card:hover { transform: translateY(-4px); }
.step-number { font-family: 'Playfair Display', serif; font-size: 2.4rem; color: #C44536; margin-bottom: 12px; }
.step-card h3 { font-size: 1.1rem; margin-bottom: 8px; color: #1C3F60; }
.step-card p { font-size: 0.9rem; color: #1C3F60; line-height: 1.5; }

/* ── BENCHMARK TABLE ── */
.bench-table {
  width: 100%; border-collapse: collapse; margin-top: 20px;
  background: #ffffff; border: 1px solid #e5e0d8; border-radius: 12px;
}
.bench-table th, .bench-table td {
  border-bottom: 1px solid #e5e0d8; padding: 14px 18px; text-align: left; font-size: 1rem; color: #1C3F60;
}
.bench-table th { font-weight: 700; background: #f9f9f9; }
.bench-table td strong { color: #1C3F60; font-weight: 700; }
.section strong, .section b, .paper-card strong { color: #1C3F60; }

/* ── FOOTER ── */
#site-footer {
  width: 100% !important; text-align: center !important; padding: 30px 5% !important;
  font-size: 0.9rem !important; color: #1C3F60 !important;
  border-top: 1px solid #e5e0d8 !important; margin-top: 60px !important;
  opacity: 1 !important; line-height: 1.6 !important;
}
#site-footer strong { color: #1C3F60 !important; }
#site-footer a { color: #C44536 !important; text-decoration: underline !important; font-weight: 600 !important; }
footer { display: none !important; }

@media (max-width: 768px) {
  .navbar { padding: 12px 5%; } .nav-links { gap: 1rem; }
  .section { padding: 40px 5% 30px; } .section-title { font-size: 2rem; }
  .hero h1 { font-size: 2.5rem; } .paper-card { padding: 20px 5%; }
}
button[id*="run"], .gradio-button.primary {
  width: 100% !important;
  background: #1C3F60 !important;
  border: 2px solid #1C3F60 !important;
  border-radius: 10px !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 1.15rem !important;
  padding: 18px 36px !important;
  cursor: pointer !important;
  letter-spacing: 0.4px !important;
  box-shadow: 0 4px 12px rgba(28,63,96,0.25) !important;
  transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
  text-transform: none !important;
}
button[id*="run"]:hover, .gradio-button.primary:hover {
  background: #152C42 !important;
  box-shadow: 0 6px 18px rgba(28,63,96,0.35) !important;
  transform: translateY(-2px);
}
button[id*="run"]:active {
  transform: scale(0.97);
}
.gr-examples label, .gr-examples .label-wrap, .gr-examples > div:first-child, .gr-examples .label-text {
  color: #1C3F60 !important;
  font-weight: 800 !important;
  font-size: 1rem !important;
  opacity: 1 !important;
  text-transform: none !important;
}
"""

# ── HTML Templates ────────────────────────────────────────────
NAV = """<nav class="navbar"><div class="nav-logo">22FAERS‑<span>NLI</span></div><ul class="nav-links"><li><a href="#validator">Validator</a></li><li><a href="#how-it-works">How it works</a></li><li><a href="#benchmark">Benchmark</a></li><li><a href="#about">About</a></li></ul></nav>"""
HERO = """<div class="section hero" id="validator"><h1 class="section-title">Validate a Signal</h1><p>Enter a drug name and an adverse event — optionally paste a PubMed abstract for instant validation, or leave it blank to search PubMed live. Built on 22 ultra‑rare neurological diseases.</p></div>"""
HOW_IT_WORKS = """<div class="section" id="how-it-works"><h2 class="section-title">How It Works</h2><div class="steps"><div class="step-card"><div class="step-number">1</div><h3>Multi‑Tier PubMed Search</h3><p>Three cascading queries retrieve relevant abstracts, even for ultra‑rare drug‑event pairs.</p></div><div class="step-card"><div class="step-number">2</div><h3>Best Evidence Selection</h3><p>All retrieved abstracts are scored; the one with the highest confidence is used.</p></div><div class="step-card"><div class="step-number">3</div><h3>Fine‑tuned PubMedBERT</h3><p>Multi‑hypothesis natural language inference scores the abstract. <strong>Displayed confidence is rescaled</strong> from the model's internal conservative score (0–50%) to the intuitive 0–100% range.</p></div><div class="step-card"><div class="step-number">4</div><h3>Validation Stamp</h3><p>Present or Absent — a final, interpretable verdict with confidence score.</p></div></div></div>"""
BENCHMARK = """<div class="section" id="benchmark"><h2 class="section-title">Benchmark Performance</h2><p style="max-width:100%;color:#1C3F60;font-size:1.05rem;line-height:1.5;">5‑system comparison on 58 manually curated evidence pairs (original gold standard). Fine‑tuned PubMedBERT matches the keyword baseline.</p><table class="bench-table"><thead><tr><th>System</th><th>Accuracy</th><th>F1‑present</th><th>F1‑absent</th></tr></thead><tbody><tr><td>Keyword co‑occurrence</td><td>0.667</td><td><strong>0.716</strong></td><td>0.596</td></tr><tr><td>DeBERTa‑v3 (zero‑shot)</td><td>0.552</td><td>0.536</td><td>0.567</td></tr><tr><td>PubMedBERT (zero‑shot)</td><td>0.526</td><td>0.571</td><td>0.471</td></tr><tr><td>Groq Llama‑3.3‑70B (few‑shot)</td><td>0.456</td><td>0.162</td><td>0.597</td></tr><tr><td><strong>PubMedBERT fine‑tuned</strong></td><td><strong>0.674</strong></td><td><strong>0.714</strong></td><td><strong>0.616</strong></td></tr></tbody></table><p style="font-size:0.9rem;color:#1C3F60;margin-top:16px;line-height:1.6;">Groq recall = 0.00–0.09 · McNemar p = 0.021 (keyword vs. zero‑shot) · Cohen's κ = 0.13 on semi‑automated labels.</p></div>"""
ABOUT = """<div class="section" id="about"><h2 class="section-title">About</h2><p style="max-width:800px;color:#1C3F60;line-height:1.8;font-size:1.05rem;"><strong>22FAERS‑NLI</strong> was developed as part of a pharmacovigilance NLP benchmark study at Delhi Technological University. The dataset contains 237 drug–adverse event pairs across 22 ultra‑rare neurological diseases, manually curated and validated against PubMed literature. This tool demonstrates that fine‑tuned biomedical language models can automate the tedious literature‑review step in pharmacovigilance pipelines.</p><p style="max-width:800px;color:#1C3F60;margin-top:16px;font-size:1rem;"><strong>Citation:</strong> Mansoori, Z.A. (2026). Automated Literature Validation for Pharmacovigilance Signals in Ultra‑Rare Neurological Diseases. Zenodo. <a href="https://doi.org/10.5281/zenodo.20359434">10.5281/zenodo.20359434</a></p></div>"""

FOOTER = """
<div id="site-footer">
  <strong>22FAERS‑NLI</strong> · Fine‑tuned PubMedBERT (F1‑present = 0.714) ·
  <a href="https://doi.org/10.5281/zenodo.20359434" target="_blank">Zenodo</a> ·
  Zaeem Ahmad Mansoori - Delhi Technological University · 2026
</div>
"""

# ── Interactive F1 Chart ─────────────────────────────────────
def create_f1_chart():
    systems = ["Keyword", "DeBERTa-v3", "PubMedBERT", "Groq (70B)", "Fine-tuned\nPubMedBERT"]
    f1_present = [0.716, 0.536, 0.571, 0.162, 0.714]
    colors = ["#3fb950", "#3d83d3", "#d5aa34", "#11c0ec", "#A30808"]
    fig = go.Figure(data=[go.Bar(
        x=systems, y=f1_present, marker_color=colors,
        text=[f"{v:.3f}" for v in f1_present], textposition="outside",
        marker_line_color="#e5e0d8", marker_line_width=1
    )])
    fig.update_layout(
        title="F1‑present for Signal Confirmation",
        template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#1C3F60"), margin=dict(t=40, b=20, l=20, r=20), height=350,
    )
    return fig

# ── Gradio App ──────────────────────────────────────────────
with gr.Blocks(title="22FAERS‑NLI") as demo:
    gr.HTML(NAV)
    gr.HTML(HERO)

    with gr.Column(elem_classes="paper-card"):
        drug_input = gr.Textbox(label="Drug Name", placeholder="e.g. cerliponase alfa")
        ae_input   = gr.Textbox(label="Adverse Event", placeholder="e.g. pleocytosis")
        abstract_input = gr.Textbox(label="PubMed Abstract (optional)", placeholder="Paste an abstract here, or leave blank to search PubMed live…", lines=4)
        run_btn    = gr.Button("Validate signal", variant="primary")

    with gr.Column(elem_classes="paper-card"):
        kw_out      = gr.Textbox(label="Keyword Co‑occurrence", interactive=False)
        model_out   = gr.Textbox(label="Model Verdict", interactive=False)
        tier_out    = gr.Textbox(label="Retrieval Info", interactive=False)
        stamp_out   = gr.HTML(label="Validation Stamp")

    with gr.Column(elem_classes="paper-card"):
        abstracts_out = gr.HTML(label="PubMed Abstracts Retrieved")

    run_btn.click(
        fn=run_validation,
        inputs=[drug_input, ae_input, abstract_input],
        outputs=[kw_out, model_out, abstracts_out, tier_out, stamp_out],
    )

    gr.Examples(
        examples=[
            ["cerliponase alfa", "seizure", "Cerliponase alfa is administered via intracerebroventricular infusion for CLN2 Batten disease. During the phase I/II trial, seizure events were recorded in 7 of 24 patients."],
            ["deferiprone", "agranulocytosis", "Deferiprone is an oral iron chelator used in transfusion‑dependent thalassaemia. Agranulocytosis is a well‑recognised idiosyncratic adverse reaction to deferiprone, with an incidence of approximately 1–2%."],
            ["pyrimethamine", "pancytopenia", "High‑dose pyrimethamine used off‑label in GM2 gangliosidosis has been associated with haematological toxicity. We report a case of pancytopenia developing after 6 months of pyrimethamine therapy, requiring dose reduction."],
            ["idursulfase", "surgery", "Idursulfase is approved for Hunter syndrome (MPS II). Patients with MPS II frequently require surgical interventions due to disease complications. These surgeries are driven by the underlying disease, not the drug."],
        ],
        inputs=[drug_input, ae_input, abstract_input],
    )

    gr.HTML(HOW_IT_WORKS)
    gr.HTML(BENCHMARK)
    gr.Plot(create_f1_chart, label="F1‑present Comparison")
    gr.HTML(ABOUT)
    gr.HTML(FOOTER)

if __name__ == "__main__":
    demo.launch(css=CSS)