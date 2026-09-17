# ✈️ FlightGuard Autonomous Systems

## 📋 Executive Summary \& System Objective

**FlightGuard Autonomous Systems** is a safety-critical aerospace telemetry and predictive maintenance platform designed to prevent catastrophic equipment failures. By integrating a robust **Tabular Stacking Ensemble Classifier**, an advanced **Remaining Useful Life (RUL) Regression Engine**, **Explainable AI (XAI)** feature attribution modules, and a **Multi-Agent Swarm Workflow**, the platform delivers automated, transparent, and legally defensible maintenance audits with precise failure timeline estimations.

## 🏗️ Modular Project Architecture

```text
flightguard\_autonomous\_systems/
├── data/
│   └── aircraft\_telemetry\_dataset.csv     # Sanitized aerospace telemetry dataset (17,107 rows, 8 cols)
├── ml\_models/
│   ├── \_\_init\_\_.py
│   ├── stacking\_ensemble\_engine.py        # Stacking Classifier for failure risk classification
│   ├── rul\_estimator.py                   # Remaining Useful Life (RUL) regression timeline engine
│   └── xai\_explainer.py                   # Local feature attribution and risk score computation engine
├── agents/
│   ├── \_\_init\_\_.py
│   ├── telemetry\_agent.py                 # Real-time IoT sensor anomaly \& failure risk monitoring agent
│   ├── xai\_audit\_agent.py                 # Translates SHAP-style attribution scores into human-readable logs
│   └── orchestrator.py                    # Master multi-agent coordination swarm pipeline
├── output/
│   ├── flightguard\_audit\_report.pdf       # Professional multi-page ReportLab compliance audit report
│   └── graphs/                            # Enterprise customized visual analytics charts (8 plots)
├── main.py                                # Master execution script integrating ML, XAI, RUL, and Swarm
├── generate\_all\_graphs.py                 # Advanced visual analytics dashboard generator script
├── tests.py                               # Unit and integration test suite
└── requirements.txt                       # Project dependencies
```

## ⚙️ Core Predictive Maintenance \& RUL Specifications

* **Remaining Useful Life (RUL) Engine:** Combines airframe operating hours, live turbine vibration ($	ext{Hz}$), exhaust gas temperature ($	ext{°C}$), and stacking ensemble risk probabilities to estimate exact operational lifespan remaining before overhaul.
* **Maintenance Window Directives:** Automatically categorizes components into actionable, high-priority timelines:

  * 🔴 **Immediate Action:** $< 48$ Hours
  * 🟡 **Schedule Within 2 Weeks**
  * 🔵 **Routine Next Check**
  * 🟢 **Nominal / Healthy**

## 🚀 End-to-End Execution Workflow

Follow these steps to run the complete pipeline locally:

1. **Clone the Repository:**

```bash
   git clone https://github.com/pushkars17/flightguard-autonomous-systems.git
   cd flightguard-autonomous-systems
   ```

2. **Install Dependencies:**

```bash
   pip install -r requirements.txt
   ```

3. **Run System Integrity \& Unit Tests:**

```bash
   python tests.py
   ```

4. **Execute Master AI Pipeline:**

```bash
   python main.py
   ```

5. **Generate Enterprise Visual Dashboards \& Audit Reports:**

```bash
   python generate\_all\_graphs.py
   ```

## 📊 Outputs \& Deliverables

* **PDF Compliance Audit Report:** Automatically compiled via ReportLab at `output/flightguard\_audit\_report.pdf`.
* **Telemetry Visualizations:** High-resolution diagnostic charts and feature attribution graphs located under `output/graphs/`.

## 💡 Tech Stack \& Libraries

* **Core Language:** Python
* **Machine Learning \& Ensembles:** Scikit-learn, NumPy, Pandas
* **Explainable AI (XAI):** Custom attribution modules \& feature scoring engines
* **Multi-Agent Orchestration:** Custom Python Agent Swarm Architecture
* **Reporting \& Visualization:** ReportLab, Matplotlib, Seaborn

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for more information.

