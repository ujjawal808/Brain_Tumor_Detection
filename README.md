<div align="center">

# 🧠 NeuroScan

### AI-Powered Brain Tumor Detection from MRI Scans

<p>
  <strong>A deep-learning powered web application for multi-class brain tumor classification using MRI images.</strong>
</p>

<p>
  Upload an MRI scan → Validate the image → Predict the tumor class → View confidence scores → Save scan history
</p>

<p>
  <a href="https://github.com/ujjawal808/Brain_Tumor_Detection">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Computer%20Vision-MRI%20Analysis-6366F1?style=flat-square"/>
  <img src="https://img.shields.io/badge/Classification-4%20Classes-10B981?style=flat-square"/>
  <img src="https://img.shields.io/badge/Authentication-Session%20Based-8B5CF6?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Active-22C55E?style=flat-square"/>
</p>

</div>

---

## 📌 Overview

**NeuroScan** is a full-stack deep-learning web application designed to classify brain MRI scans into four categories:

* 🧬 **Glioma**
* 🧠 **Meningioma**
* 🩺 **Pituitary Tumor**
* ✅ **No Tumor**

The application combines **deep learning, image preprocessing, MRI-specific validation, confidence filtering, user authentication, and database-backed scan history** into a single web interface.

Instead of directly sending every uploaded image to the tumor classifier, NeuroScan first performs lightweight image checks to identify images that are unlikely to be brain MRI scans. Valid scans are then processed by the trained classification model.

### 🎯 Project Goal

The primary goal of NeuroScan is to demonstrate how a trained deep-learning image classification model can be integrated into a practical web application with:

> **Image Upload → Validation → AI Prediction → Confidence Analysis → Database Storage → Visualization**

---

## ⚠️ Medical Disclaimer

> **NeuroScan is an educational and research project.**
>
> It is **not a medical diagnostic system** and should not be used as a substitute for a qualified radiologist, neurologist, or other healthcare professional.
>
> Model predictions may be incorrect and should not be used to make medical decisions.

---

# ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧠 AI Classification

Classifies MRI scans into **4 categories** using a trained deep-learning model.

### 🔍 MRI Validation

Performs lightweight image checks before classification to reject obvious non-MRI or unsuitable images.

### 📊 Confidence Analysis

Returns confidence scores for all four classes and applies a confidence threshold before accepting a prediction.

### 🔄 Test-Time Augmentation

Uses multiple model predictions and averages the results to provide more stable prediction scores.

</td>

<td width="50%">

### 🔐 User Authentication

Signup/login system with password hashing using Werkzeug.

### 🗄️ Scan History

Stores user scan results, predictions, confidence values, and timestamps in SQLite.

### 📈 Personal Dashboard

Displays total scans, tumor detections, clear scans, and recent scan history.

### 🎨 Modern Dark UI

Responsive dark-themed interface with drag-and-drop image upload and dynamic prediction visualization.

</td>
</tr>
</table>

---

# 🧩 System Architecture

```text
                         ┌─────────────────────┐
                         │       User          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Upload MRI Scan   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │     Image Preprocessing      │
                    │       PIL + NumPy            │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     MRI Validation Layer     │
                    │                              │
                    │ • Aspect Ratio               │
                    │ • Color / Grayscale Check    │
                    │ • Dark Background Check      │
                    │ • Uniform Image Check        │
                    │ • Brightness Check            │
                    └──────────────┬───────────────┘
                                   │
                            Valid MRI?
                           /           \
                         No             Yes
                         │               │
                         ▼               ▼
                   Reject Image    ┌─────────────────┐
                                   │ Tumor Classifier│
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ TTA Prediction  │
                                   │ Multiple Runs   │
                                   │ + Averaging     │
                                   └────────┬────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │ Confidence Check │
                                  └────────┬─────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  Prediction + Scores   │
                              └────────────┬───────────┘
                                           │
                              ┌────────────┴────────────┐
                              ▼                         ▼
                     ┌─────────────────┐       ┌─────────────────┐
                     │ SQLite Database │       │ Web Dashboard   │
                     │   Scan History  │       │ Results + Stats │
                     └─────────────────┘       └─────────────────┘
```

---

# 🧪 Supported Tumor Classes

|     Class    | Display Label          | Description                                    |
| :----------: | ---------------------- | ---------------------------------------------- |
|   `glioma`   | 🧬 **Glioma**          | Tumor originating from glial cells.            |
| `meningioma` | 🧠 **Meningioma**      | Tumor arising from the meninges.               |
|   `notumor`  | ✅ **No Tumor**         | No tumor detected by the classification model. |
|  `pituitary` | 🩺 **Pituitary Tumor** | Tumor located around the pituitary region.     |

---

# 🔬 MRI Validation Pipeline

One of the important additions to the application is an image validation layer that runs **before the tumor classifier**.

The validation function performs several lightweight checks.

### 1️⃣ Aspect Ratio Check

Images that are extremely tall or wide are rejected because they may not resemble the expected MRI input format.

```python
ratio = height / width

if ratio > 1.6 or ratio < 0.6:
    reject_image()
```

### 2️⃣ Color Check

The application checks whether the image is strongly colored instead of approximately grayscale.

```text
RGB Image
    │
    ▼
Channel Difference Analysis
    │
    ├── High Difference → Reject
    │
    └── Low Difference → Continue
```

### 3️⃣ Dark Background Check

Brain MRI images commonly contain a significant dark background around the brain region.

### 4️⃣ Uniform Image Check

Extremely uniform images can indicate blank or corrupted files.

### 5️⃣ Brightness Check

Images that are excessively bright are rejected as potentially unsuitable MRI inputs.

---

# 🤖 Model Prediction Pipeline

After validation, the image enters the tumor classification stage.

```text
MRI Image
    │
    ▼
Resize to 260 × 260
    │
    ▼
EfficientNetV2 Preprocessing
    │
    ▼
Tumor Classification Model
    │
    ▼
Multiple Predictions
    │
    ▼
Average Predictions
    │
    ▼
4-Class Probability Scores
    │
    ▼
Highest Probability Class
```

The current application uses **Test-Time Augmentation-style repeated inference** by running the model multiple times and averaging the resulting prediction vectors.

```python
scores = [
    model.predict(arr, verbose=0)[0]
    for _ in range(5)
]

final_score = np.mean(scores, axis=0)
```

---

# 📊 Confidence Threshold

After classification, NeuroScan checks the highest prediction confidence.

The current tumor confidence threshold is:

```python
TUMOR_CONFIDENCE = 0.60
```

If the confidence is below the threshold, the application asks the user to provide a clearer or different MRI scan rather than presenting a low-confidence classification as a normal result.

---

# 🛠️ Technology Stack

## Backend

| Technology                | Purpose                                 |
| ------------------------- | --------------------------------------- |
| 🐍 **Python**             | Core programming language               |
| 🌶️ **Flask**             | Web application framework               |
| 🗃️ **Flask-SQLAlchemy**  | Database ORM                            |
| 🤖 **TensorFlow / Keras** | Deep-learning model inference           |
| 🖼️ **Pillow**            | Image loading and preprocessing         |
| 🔢 **NumPy**              | Numerical and image-array operations    |
| 🔐 **Werkzeug**           | Password hashing and security utilities |

## Frontend

| Technology | Purpose                          |
| ---------- | -------------------------------- |
| HTML5      | Page structure                   |
| CSS3       | UI and responsive styling        |
| JavaScript | Dynamic interactions             |
| Jinja2     | Server-side templating           |
| Fetch API  | Asynchronous prediction requests |
| Canvas     | Prediction visualization         |

## Database

**SQLite**

Main entities:

```text
User
 ├── id
 ├── name
 ├── email
 ├── password_hash
 └── created_at

Scan
 ├── id
 ├── user_id
 ├── filename
 ├── prediction
 ├── label
 ├── confidence
 ├── is_tumor
 └── scanned_at
```

---

# 📁 Project Structure

```text
Brain_Tumor_Detection/
│
├── Brain_Tumor_Detection/
│   │
│   ├── app.py
│   │
│   ├── main.py
│   │
│   ├── Templates/
│   │   ├── dashboard.html
│   │   ├── login.html
│   │   └── Signup.html
│   │
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   │
│   └── instance/
│       └── neuroscan.db
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The trained model file is not included in the repository if it is excluded by `.gitignore` or GitHub file-size limitations. Configure the model path in `app.py` according to your local environment.

---

# 🚀 Getting Started

## Prerequisites

Make sure you have:

* Python **3.10+**
* pip
* Git
* A compatible trained Keras model
* Windows, macOS, or Linux

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/ujjawal808/Brain_Tumor_Detection.git
```

Navigate into the project:

```bash
cd Brain_Tumor_Detection
cd Brain_Tumor_Detection
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure the Model

Place your trained model in the appropriate location and configure the path used by `app.py`.

For example:

```python
tumor_model = load_model(
    r"path/to/brain_tumor_model_v2.h5"
)
```

### Recommended Improvement

For portability, avoid hard-coded Windows paths and use a project-relative path:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "brain_tumor_model_v2.h5"

tumor_model = load_model(MODEL_PATH)
```

This makes the application easier to run on other machines.

---

## 5️⃣ Run the Application

```bash
python app.py
```

The Flask development server will start.

Open:

```text
http://127.0.0.1:5000
```

---

# 🖥️ Application Flow

### 🔐 Step 1 — Create an Account

Register using:

```text
Name
Email
Password
```

Passwords are stored using hashing rather than plain text.

---

### 🔑 Step 2 — Login

Authenticate using your registered email and password.

---

### 📤 Step 3 — Upload MRI

Upload a supported image file through the dashboard.

```text
JPG / JPEG / PNG
```

---

### 🔍 Step 4 — Image Validation

The application performs preliminary MRI-specific checks.

```text
Is image suitable?
       │
   ┌───┴───┐
   │       │
  NO      YES
   │       │
Reject    Model
```

---

### 🧠 Step 5 — AI Prediction

The classification model predicts one of the four classes.

---

### 📊 Step 6 — View Results

The dashboard displays:

* Predicted class
* Confidence percentage
* Scores for all classes
* Tumor / No Tumor status
* Prediction description

---

### 🗄️ Step 7 — Save Scan

The result is stored in SQLite and becomes part of the user's scan history.

---

# 📡 API Reference

## `POST /predict`

Predicts the class of an uploaded MRI scan.

### Authentication

A valid logged-in session is required.

### Request

```text
Content-Type: multipart/form-data
```

| Parameter | Type  | Required | Description          |
| --------- | ----- | :------: | -------------------- |
| `file`    | Image |     ✅    | MRI image to analyze |

### Example Response

```json
{
  "prediction": "glioma",
  "label": "Glioma",
  "confidence": 94.3,
  "all_scores": {
    "glioma": 94.3,
    "meningioma": 3.1,
    "notumor": 1.4,
    "pituitary": 1.2
  },
  "description": "A malignant tumor originating in glial cells.",
  "is_tumor": true
}
```

---

# 🔐 Security Features

NeuroScan includes several basic application-level security measures:

### Password Hashing

Passwords are hashed using Werkzeug:

```python
generate_password_hash(password)
```

Authentication is verified using:

```python
check_password_hash(...)
```

### Session-Based Authentication

Protected pages require a valid user session.

```python
@login_required
def dashboard():
    ...
```

### User-Specific Scan History

Scans are associated with the authenticated user's ID.

```text
User
 │
 ├── Scan 1
 ├── Scan 2
 ├── Scan 3
 └── Scan 4
```

---

# 📈 Dashboard

The dashboard provides a quick overview of the user's activity.

### Statistics

```text
┌─────────────────┐
│   Total Scans   │
├─────────────────┤
│       25        │
└─────────────────┘

┌─────────────────┐
│ Tumor Detected  │
├─────────────────┤
│       14        │
└─────────────────┘

┌─────────────────┐
│   No Tumor      │
├─────────────────┤
│       11        │
└─────────────────┘
```

The recent scan history shows the latest predictions along with confidence and timestamps.

---

# 🧪 Example Prediction Flow

```text
Uploaded Image
      │
      ▼
┌───────────────┐
│ MRI Validation│
└───────┬───────┘
        │
        ▼
┌────────────────────┐
│ Tumor Classification│
└─────────┬──────────┘
          │
          ▼
     Prediction
          │
     ┌────┴─────┐
     ▼          ▼
   Tumor      No Tumor
     │          │
     └────┬─────┘
          ▼
   Confidence Score
          │
          ▼
     Save to DB
          │
          ▼
   Display Results
```

---

# 🧮 Complexity Overview

| Operation           |       Approximate Complexity |
| ------------------- | ---------------------------: |
| Image loading       |                         O(P) |
| Image preprocessing |                         O(P) |
| Pixel validation    |                         O(P) |
| Model inference     |              Model-dependent |
| TTA inference       |       O(N × Model Inference) |
| Database lookup     |      O(log n) / DB-dependent |
| Dashboard rendering | O(k), where k = recent scans |

Where **P** represents the number of processed image pixels.

---

# 📚 Learning Objectives

This project demonstrates practical experience with:

* 🧠 Deep Learning
* 🖼️ Image Classification
* 👁️ Computer Vision
* 🐍 Python
* 🌶️ Flask
* 🤖 TensorFlow / Keras
* 🔢 NumPy
* 🖼️ Pillow
* 🗃️ SQL / SQLite
* 🔐 Authentication
* 🌐 REST-style APIs
* 🎨 Frontend development
* 📊 Confidence analysis
* 🔄 Test-Time Augmentation
* 🧪 Input validation
* 🔗 Full-stack application integration

---

# 🚧 Future Improvements

The project can be extended with:

* [ ] Replace hard-coded model paths with environment variables
* [ ] Add model evaluation metrics dashboard
* [ ] Add confusion matrix visualization
* [ ] Add precision, recall and F1-score reporting
* [ ] Add Grad-CAM heatmaps for model explainability
* [ ] Add downloadable prediction reports
* [ ] Add stronger image-type validation
* [ ] Add REST API authentication
* [ ] Add Docker support
* [ ] Deploy the application to a cloud platform
* [ ] Add automated unit and integration tests
* [ ] Improve responsive mobile UI
* [ ] Add model version tracking

---

# 🧪 Testing Checklist

Before deployment, test the following:

```text
Authentication
├── Signup
├── Login
├── Incorrect password
└── Logout

Image Upload
├── Valid MRI
├── Invalid image
├── Unsupported file
├── Blank image
└── Non-MRI image

Prediction
├── Glioma
├── Meningioma
├── No Tumor
├── Pituitary
└── Low-confidence image

Database
├── Scan saved
├── User-specific history
└── Dashboard statistics
```

---

# 🐛 Troubleshooting

### `ModuleNotFoundError`

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

### Model Not Found

Check the model path configured in:

```text
Brain_Tumor_Detection/app.py
```

---

### Port Already in Use

Run Flask on another port:

```python
app.run(debug=True, port=5001)
```

Then open:

```text
http://127.0.0.1:5001
```

---

### TensorFlow / Keras Compatibility

Make sure the installed TensorFlow, Keras, Python version, and trained model format are compatible.

Check:

```bash
python --version
pip show tensorflow
pip show keras
```

---

# 🤝 Contributing

Contributions are welcome!

### 1. Fork the repository

```bash
git clone https://github.com/ujjawal808/Brain_Tumor_Detection.git
```

### 2. Create a feature branch

```bash
git checkout -b feature/amazing-feature
```

### 3. Commit your changes

```bash
git add .
git commit -m "Add amazing feature"
```

### 4. Push your branch

```bash
git push origin feature/amazing-feature
```

### 5. Open a Pull Request

Explain what you changed and why.

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

# 👨‍💻 Author

<div align="center">

### Ujjawal Baliyan

<p>
  <a href="https://github.com/ujjawal808">
    <img src="https://img.shields.io/badge/GitHub-Ujjawal%20Baliyan-181717?style=for-the-badge&logo=github"/>
  </a>
</p>

<p>
  <strong>MCA Student • Software Development • AI/ML • Data Structures & Algorithms</strong>
</p>

</div>

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

<br/>

**Built with 🧠 Deep Learning + 🐍 Python + 🌶️ Flask + 🤖 TensorFlow**

<br/>

<sub>
For educational and research purposes only. Not intended for clinical diagnosis.
</sub>

</div>
