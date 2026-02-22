# 💰 MLOps Prediction Profit

End-to-end Machine Learning project for predicting business profit, built using a reproducible MLOps pipeline with DVC, Docker, and CI/CD integration.

---

## 📌 Project Overview

This project aims to predict business profit using structured data and a supervised machine learning approach.

The system is designed following modern MLOps best practices:

- 📦 Data versioning with DVC  
- ⚙️ Parameter tracking with YAML  
- 🔁 Reproducible pipeline automation  
- 📊 Model evaluation tracking  
- 🐳 Docker containerization  
- 🔄 CI integration using GitHub Actions  

This repository demonstrates a production-oriented ML workflow rather than just notebook experimentation.

---

## 🏗️ Project Architecture

**Data → Preprocessing → Training → Evaluation → Model Saving → API Serving → Dockerized Deployment**

Main components used:

- **DVC** — Dataset & pipeline versioning  
- **Scikit-learn** — Machine learning modeling  
- **FastAPI** — Model serving API  
- **Docker** — Containerization  
- **GitHub Actions** — Continuous Integration  

---

## 📂 Project Structure

```bash
.
├── data/                # Raw and processed datasets (tracked by DVC)
├── model/               # Trained model artifacts
├── src/                 # Training & preprocessing scripts
├── app.py               # FastAPI application
├── dvc.yaml             # Pipeline definition
├── params.yaml          # Hyperparameters configuration
├── metrics.json         # Model evaluation results
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── .github/workflows/   # CI configuration
```

---

## ⚙️ Machine Learning Pipeline

The pipeline is defined in `dvc.yaml` and includes:

- Data preprocessing  
- Model training  
- Model evaluation  
- Metric tracking  

To reproduce the full pipeline:

```bash
dvc repro
```

To view model performance metrics:

```bash
dvc metrics show
```

---

## 📊 Model Performance

Model evaluation results are stored in:

```
metrics.json
```

## 🚀 Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open API documentation:

```
http://localhost:8000/docs
```

---

## 🐳 Run with Docker

Build Docker image:

```bash
docker build -t mlops-profit .
```

Run container:

```bash
docker run -p 8000:8000 mlops-profit
```

---

## 🔁 CI/CD Integration

This project includes GitHub Actions workflow to:

- Build Docker image
- Validate dependencies
- Ensure pipeline reproducibility

---

## 🧠 Key MLOps Concepts Implemented

- Reproducible ML pipeline  
- Data versioning (DVC)  
- Parameterized experimentation  
- Metric tracking  
- Containerized model serving  
- API-based inference  
- Continuous Integration  

---

## 🎯 Future Improvements

- Add model monitoring  
- Add model drift detection  
- Deploy to cloud environment  
- Implement automated retraining pipeline  
