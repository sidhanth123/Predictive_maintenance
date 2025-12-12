🚀 Predictive Maintenance — Machine Learning + Streamlit Dashboard
This project provides an end-to-end predictive maintenance system built using:
Machine Learning (RandomForest, XGBoost)
Full preprocessing + training pipeline
SHAP explainability
Streamlit interactive UI dashboard
Batch prediction support
Clean production-ready folder structure

📂 Project Structure
Predictive Maintenance/
│── app.py               # Streamlit dashboard
│── train.py             # Model training (pipelines + saves model)
│── predict.py           # Run inference on new data
│── utils.py             # Data cleaning + preprocessing helper
│
├── data/
│   └── predictive_maintenance.csv
│
├── models/
│   ├── best_model.pkl
│   ├── RandomForest_pipeline.pkl
│   ├── XGBoost_pipeline.pkl
│
├── requirements.txt
└── README.md

🛠 Installation
pip install -r requirements.txt

▶️ Run the Streamlit Dashboard
streamlit run app.py

⚙️ Train the Model
python train.py

🔍 Predict New Data
python predict.py