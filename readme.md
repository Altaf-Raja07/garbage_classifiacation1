# Garbage Classification

A web-based application that classifies types of waste (garbage) from images, using a trained machine learning model. Built with Python and displayed via a simple web interface.

---

## 🚀 Features

- **Predict** the type of waste from user-uploaded images.
- **Modular design**, allowing the classification model to be updated separately from the interface.
- **HTML templates** for a clean and interactive front-end.

---

## 📂 Repository Layout

| Path                           | Description                                      |
|--------------------------------|--------------------------------------------------|
| `app.py`                       | Main application entry point (e.g. Flask app).   |
| `util.py`                      | Utility functions (e.g. image preprocessing).    |
| `waste_classification_model.ipynb` | Notebook to train or evaluate the classification model. |
| `model/`                       | Trained model files for inference.               |
| `static/images/`               | Supporting images (examples, assets, etc.).      |
| `templates/`                   | HTML templates for the front-end interface.      |

---

## 🛠️ Prerequisites

- Python ≥ 3.x  
- Pip or Conda package manager

Recommended virtual environment setup:
```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Clone the repository
```bash
git clone https://github.com/Altaf-Raja07/garbage_classifiacation1.git
cd garbage_classifiacation1
```

## 📊 Model Training & Evaluation

- Use the `waste_classification_model.ipynb` notebook to:
  - Train a new model  
  - Evaluate model performance  
  - Experiment with datasets or parameters
