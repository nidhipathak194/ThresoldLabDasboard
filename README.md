# ⚖️ Threshold Lab: Sorites Paradox & Fraud Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uniprojectfinal.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-nidhipathak194%2FUniprojectFinal-blue?logo=github)](https://github.com/nidhipathak194/UniprojectFinal)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive, executive-grade decision intelligence platform investigating the **Sorites Paradox** in machine learning threshold optimization for credit card fraud detection.

---

## 🌐 Live Public Deployment

The application is deployed live on **Streamlit Community Cloud**:
* **Live App URL**: [https://uniprojectfinal.streamlit.app](https://uniprojectfinal.streamlit.app)
* **GitHub Repository**: [https://github.com/nidhipathak194/UniprojectFinal](https://github.com/nidhipathak194/UniprojectFinal)

---

## 🚀 Public Deployment Guide (Streamlit Community Cloud)

To deploy or update this project on Streamlit Community Cloud:

1. **Fork or Push to GitHub**:
   Ensure the latest code is committed and pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Configure Streamlit Community Cloud deployment"
   git push origin main
   ```

2. **Connect to Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io).
   - Sign in with your GitHub account (`nidhipathak194`).
   - Click **New app**.

3. **Configure App Settings**:
   - **Repository**: `nidhipathak194/UniprojectFinal`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL custom alias**: `uniprojectfinal` (or standard `uniprojectfinal.streamlit.app`)

4. **Deploy**:
   - Click **Deploy!**. Streamlit will automatically read `requirements.txt` and `.streamlit/config.toml`, build dependencies, and launch the application.

---

## 🐳 Docker Deployment Guide

The project includes containerization setup via `Dockerfile`.

### Build & Run Container Locally
```bash
# 1. Build the Docker image
docker build -t threshold-lab-app .

# 2. Run the container on port 8501
docker run -p 8501:8501 threshold-lab-app
```
Access the containerized app at `http://localhost:8501`.

---

## 💻 Local Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/nidhipathak194/UniprojectFinal.git
cd UniprojectFinal

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run unit tests
python3 -m unittest discover tests

# 5. Launch Streamlit dashboard
streamlit run app.py
```

---

## 📊 Theoretical Framework

### 1. The Sorites Paradox in Machine Learning
The Sorites Paradox (Vagueness Problem) explores how infinitesimal boundary shifts lead to macroscopic decision shifts. In fraud classification:
- Is a transaction with fraud probability \( p(x) = 0.0501 \) fundamentally different from one with \( p(x) = 0.0499 \)?
- Small changes in operational threshold \( \tau \) reclassify thousands of transactions between "Approved" and "Flagged for Audit".

### 2. Elkan's Cost-Sensitive Optimal Threshold
According to Elkan (2001), the theoretical expected-cost-minimizing threshold \( \tau^* \) is given by:
\[
\tau^* = \frac{C(FP)}{C(FP) + C(FN)}
\]
Where:
- \( C(FP) \): Cost of False Positive (auditing/friction fee for a legitimate customer)
- \( C(FN) \): Cost of False Negative (uncaptured fraud transaction amount)

---

## 📁 Repository Structure

```
UniprojectFinal/
├── app.py                      # Main Executive Streamlit Dashboard
├── Dockerfile                  # Container definition for containerized deployment
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Production theme and server configuration
├── src/
│   ├── data_loader.py          # Synthetic dataset generator, uploaded-CSV cleaning/validation & preprocessing
│   ├── ml_engine.py            # Fraud ML classifiers & probability calibration (Platt/Isotonic)
│   └── sorites_engine.py       # Sorites paradox analysis & Elkan cost curve calculations
└── tests/
    └── test_engines.py         # Automated unit test suite
```

---

## 📥 Uploading Your Own CSV

Selecting **"Upload Custom CSV"** in the sidebar runs any file you provide
through a validation and cleaning step (`load_custom_csv` in
`src/data_loader.py`) before it reaches the ML pipeline:

1. Column names are matched to `Time`, `Amount`, `V1`–`V28`, `Class`
   case-insensitively, so minor header differences don't cause a failure.
2. Any columns beyond those required (e.g. a stray index column some CSV
   exporters add) are dropped.
3. All feature columns are coerced to numeric and `Class` is coerced to a
   binary 0/1 label; rows that don't survive coercion are removed.
4. Exact duplicate rows are removed (the authentic ULB dataset itself
   contains 1,081 of these).
5. A **Data Processing Report** — rows read, rows kept, duplicates
   removed, invalid rows removed, resulting class balance — is shown in
   an expander in the sidebar so the cleaning step is transparent, not
   silent.

If the file is missing a required column, or has no fraud examples left
after cleaning, the app shows a clear error and falls back to the
synthetic generator rather than crashing.

**Performance note:** the full authentic dataset (284,807 rows) works
correctly end-to-end, but the app eagerly trains all three classifiers
across all three calibration methods on first load — on the full dataset
this can take several minutes for the tree-based models (Random Forest /
XGBoost), even though it's cached afterwards via `@st.cache_resource`. The
Synthetic Sample Size slider is the faster path for interactive demos.

---

## 📄 License
This project is released under the MIT License.
