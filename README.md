<div align="center">

<h1>🧠 NeuroScan — Brain Tumor Detection</h1>

<p>
  A deep-learning powered web application that detects brain tumors from MRI scans in seconds.<br/>
  Upload a scan, get an instant prediction with confidence scores — no radiologist waiting room required.
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-3.15-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Accuracy-High-34d399?style=flat-square"/>
  <img src="https://img.shields.io/badge/Classes-4%20Tumor%20Types-4f8ef7?style=flat-square"/>
  <img src="https://img.shields.io/badge/Database-SQLite-003B57?style=flat-square&logo=sqlite"/>
  <img src="https://img.shields.io/badge/Auth-Session%20Based-7c3aed?style=flat-square"/>
</p>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [How It Works](#-how-it-works)
- [Tumor Classes](#-tumor-classes)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔬 Overview

**NeuroScan** is a full-stack medical imaging assistant that uses a Convolutional Neural Network (CNN) trained on MRI brain scans to classify tumors into four categories. It wraps the model in a sleek, dark-themed web app with user authentication, scan history, and real-time confidence visualization.

> ⚠️ **Disclaimer:** This tool is intended for educational and research purposes only. It is **not** a substitute for professional medical diagnosis.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧬 **4-Class Classification** | Detects Glioma, Meningioma, Pituitary Tumor, or No Tumor |
| 📊 **Confidence Visualization** | Animated donut chart + score bars for all 4 classes |
| 🔐 **User Authentication** | Secure signup/login with hashed passwords (Werkzeug) |
| 🗄️ **Scan History** | Every scan is saved to SQLite — last 5 shown on dashboard |
| 📈 **Personal Stats** | Total scans, tumor count, clear scan count per user |
| 🖼️ **Drag & Drop Upload** | Drop a JPG/PNG MRI scan directly onto the upload zone |
| 🌙 **Dark UI** | Minimal dark-mode interface built with plain CSS |
| ⚡ **Real-time Results** | Async fetch — no page reload needed |

---

## 🛠️ Tech Stack

### Backend
- **Python 3.12** — Core language
- **Flask 3.1** — Web framework & routing
- **Flask-SQLAlchemy** — ORM for SQLite database
- **TensorFlow 2.21 / Keras 3.15** — Deep learning model inference
- **Werkzeug** — Password hashing & WSGI utilities
- **Pillow** — MRI image preprocessing

### Frontend
- **Jinja2** — Server-side HTML templating
- **Vanilla JS** — Async fetch, drag-and-drop, Canvas donut chart
- **CSS Custom Properties** — Dark theme design system (no framework needed)

### Database
- **SQLite** — Zero-config embedded database
  - `User` table — name, email, password hash, created_at
  - `Scan` table — prediction, confidence, tumor flag, timestamp

---

## 📁 Project Structure

```
Brain_Tumor_Detection/
├── app.py                    # Flask app — routes, auth, prediction logic
├── main.py                   # PyCharm entry point placeholder
├── model/
│   └── brain_tumor_model.keras   # Trained CNN model (Keras format)
├── Templates/
│   ├── dashboard.html        # Main scan upload + result UI
│   ├── login.html            # Sign-in page
│   └── Signup.html           # Registration page
├── instance/
│   └── neuroscan.db          # SQLite database (auto-created)
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- pip
- Git

### 1 — Clone the repository

```bash
git clone https://github.com/ujjawal808/Brain_Tumor_Detection.git
cd Brain_Tumor_Detection/Brain_Tumor_Detection
```

### 2 — Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Add the model file

Download or place your trained model file at:

```
Brain_Tumor_Detection/model/brain_tumor_model.keras
```

Then update line 47 of [`app.py`](Brain_Tumor_Detection/app.py) to point to the correct path:

```python
model = load_model("model/brain_tumor_model.keras")
```

### 5 — Run the app

```bash
python app.py
```

Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## ⚙️ How It Works

```
User uploads MRI  ──▶  Flask /predict  ──▶  PIL resize to 180×180
                                                     │
                                              TF model.predict()
                                                     │
                                        softmax scores for 4 classes
                                                     │
                              ┌──────────────────────┴────────────────────────┐
                              │  Save Scan to DB   │  Return JSON to browser  │
                              └────────────────────┴────────────────────────  ┘
                                                     │
                                        Animated donut chart + bars
```

1. The uploaded image is read into memory (no disk writes).
2. It is converted to RGB and resized to **180 × 180 px**.
3. The Keras CNN outputs a **softmax probability vector** over 4 classes.
4. The top class and its confidence are returned as JSON.
5. The result is saved to the user's scan history in SQLite.
6. The frontend renders the donut chart and score bars in real time.

---

## 🧪 Tumor Classes

| Class | Label | Description |
|---|---|---|
| `glioma` | **Glioma** | Malignant tumor originating in glial cells. Requires immediate specialist review. |
| `meningioma` | **Meningioma** | Tumor arising from the meninges. Usually benign and slow-growing. |
| `notumor` | **No Tumor** | No tumor tissue detected. Brain tissue appears structurally normal. |
| `pituitary` | **Pituitary Tumor** | Tumor at the base of the brain. Usually benign but may affect hormones. |

---

## 📡 API Reference

### `POST /predict`

Requires an active session (logged-in user).

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `image/*` | Brain MRI scan (JPG or PNG) |

**Response** — `application/json`

```json
{
  "prediction":  "glioma",
  "label":       "Glioma",
  "confidence":  94.3,
  "all_scores":  { "glioma": 94.3, "meningioma": 3.1, "notumor": 1.4, "pituitary": 1.2 },
  "description": "A malignant tumor originating in glial cells...",
  "is_tumor":    true
}
```

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Made with ❤️ by <a href="https://github.com/ujjawal808">Ujjawal Baliyan</a></p>
  <p><sub>For educational and research purposes only. Not for clinical use.</sub></p>
</div>
