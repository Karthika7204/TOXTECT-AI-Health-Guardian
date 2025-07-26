# 🛡️ TOXTECT: AI Health Guardian

**TOXTECT: AI Health Guardian** is an AI-powered health management platform designed to assist users in tracking, predicting, and analyzing various aspects of personal health using intelligent modules. This all-in-one system blends machine learning, image processing, and health diagnostics to provide proactive wellness support.

---

## 🚀 Features

The system consists of multiple AI modules:

| 🧠 Module | 🔍 Description |
|----------|----------------|
| **Food Scanner** | Scans packaged food for allergens and harmful ingredients using OCR and a trained ML model |
| **Tablet Tracker** | Keeps track of medications and sends alerts for dosage timing |
| **Vision Checker** | Helps in vision self-assessment using visual acuity tests |
| **Cosmetics Scanner** | Detects harmful chemicals in cosmetic product ingredients |
| **Mind Mate** | Mental wellness companion with AI-generated prompts and mood tracking |
| **Symptoms Prediction** | Predicts possible health conditions based on entered symptoms |
| **BMI Food Planner** | Suggests personalized meal plans based on BMI classification |
| **Heart Attack Prediction** | Uses health metrics to predict the likelihood of heart attacks |
| **Calories Burn Predictor** | Estimates calories burned based on food intake and physical activity |

---

## 🧠 Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Django
- **ML Tools:** EasyOCR, Pandas, Sklearn, PyTorch
- **Deployment:** Localhost / GitHub (development stage)

---

## 🛠 How to Run the Project Locally

```bash
# Clone the repo
git clone https://github.com/Karthika7204/TOXTECT-AI-Health-Guardian.git
cd TOXTECT-AI-Health-Guardian

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the server
python manage.py runserver
