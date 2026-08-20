import io
import numpy as np
from datetime import datetime
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "super_secret_medical_key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///neuroscan.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── Database models ───────────────────────────────────────────────────────────

class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    scans         = db.relationship('Scan', backref='user', lazy=True)

class Scan(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename    = db.Column(db.String(256))
    prediction  = db.Column(db.String(50))
    label       = db.Column(db.String(100))
    confidence  = db.Column(db.Float)
    is_tumor    = db.Column(db.Boolean)
    scanned_at  = db.Column(db.DateTime, default=datetime.utcnow)

# ── Load models ───────────────────────────────────────────────────────────────

# ── Load ML model ─────────────────────────────────────────────────────────────
tumor_model = load_model(
    r"C:\Users\ujjawal baliyan\Downloads\brain_tumor_model_v2.h5"
)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
TUMOR_SIZE = 260

# ── Thresholds ────────────────────────────────────────────────────────────────
# HOW TO TUNE THESE:
# - Lower BRAIN_CONFIDENCE if real brain MRIs are being rejected
# - Raise BRAIN_CONFIDENCE if chest/hand X-rays are still getting through
# - Lower TUMOR_CONFIDENCE if correct scans show <70% on tumor prediction
BRAIN_CONFIDENCE = 0.80   # stage 1: must be this sure it's a brain MRI
TUMOR_CONFIDENCE = 0.60   # stage 2: lowered from 0.70 — brain already validated

# ── Image feature checks ──────────────────────────────────────────────────────
# These are fast pixel-level rules that catch obvious non-brain images
# BEFORE even running the model — saves time and catches edge cases

def is_likely_brain_mri(img_array):
    """
    Fast heuristic checks on raw pixel values.
    Brain MRIs have specific characteristics:
    - Mostly dark background (large black border around skull)
    - Roughly circular bright region (the brain)
    - Grayscale or near-grayscale (low color saturation)
    - Specific aspect ratio (close to square)
    Returns (is_valid, reason_if_rejected)
    """
    h, w = img_array.shape[:2]

    # CHECK 1: aspect ratio — brain MRIs are roughly square
    # Chest X-rays are usually portrait (taller than wide)
    # Hand X-rays can be very elongated
    ratio = h / w
    if ratio > 1.6 or ratio < 0.6:
        return False, f"Image shape {w}x{h} doesn't match brain MRI (too tall or wide)"

    # CHECK 2: color saturation — brain MRIs are grayscale
    # Color photos, some ultrasound images fail this
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    max_channel_diff = float(np.mean(np.abs(r.astype(float) - g.astype(float)) +
                                     np.abs(g.astype(float) - b.astype(float))))
    if max_channel_diff > 30:
        return False, f"Image appears to be in color (avg channel diff: {max_channel_diff:.1f}). Brain MRIs are grayscale."

    # CHECK 3: dark border — brain MRIs have significant black background
    # Natural photos and some X-rays fail this
    gray = np.mean(img_array, axis=2)  # convert to single channel
    dark_pixel_ratio = float(np.mean(gray < 30))  # pixels near-black
    if dark_pixel_ratio < 0.15:
        return False, f"Not enough dark background ({dark_pixel_ratio*100:.1f}%). Brain MRIs have dark borders."

    # CHECK 4: not completely uniform (blank image check)
    std = float(np.std(gray))
    if std < 10:
        return False, "Image is too uniform — may be blank or corrupted."

    # CHECK 5: reasonable brightness range — MRIs have both dark and bright regions
    bright_ratio = float(np.mean(gray > 200))
    if bright_ratio > 0.7:
        return False, "Image is too bright overall — not typical for brain MRI."

    return True, "ok"


def preprocess_for_model(img, size):
    """Resize, add batch dim, apply EfficientNet preprocessing."""
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    return arr


def predict_with_tta(model, arr, n=5):
    """Average n predictions for stable confidence (Test Time Augmentation)."""
    scores = [model.predict(arr, verbose=0)[0] for _ in range(n)]
    return np.mean(scores, axis=0)



# ── Auth helper ───────────────────────────────────────────────────────────────

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not name or not email or not password:
            flash('All fields are required.', 'error')
        elif len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
        elif User.query.filter_by(email=email).first():
            flash('An account with that email already exists.', 'error')
        else:
            user = User(name=name, email=email,
                        password_hash=generate_password_hash(password))
            db.session.add(user)
            db.session.commit()
            session['user_id']   = user.id
            session['user_name'] = user.name
            return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user     = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash('Incorrect email or password.', 'error')
        else:
            session['user_id']   = user.id
            session['user_name'] = user.name
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user  = User.query.get(session['user_id'])
    scans = (Scan.query.filter_by(user_id=user.id)
                 .order_by(Scan.scanned_at.desc())
                 .limit(5).all())
    total    = Scan.query.filter_by(user_id=user.id).count()
    tumor_ct = Scan.query.filter_by(user_id=user.id, is_tumor=True).count()
    no_tumor = Scan.query.filter_by(user_id=user.id, is_tumor=False).count()
    return render_template('dashboard.html', user=user, scans=scans,
                           total=total, tumor_ct=tumor_ct, no_tumor=no_tumor)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # ── STAGE 1: pixel-level heuristic checks ────────────────────────────
        raw_arr = np.array(img.resize((260, 260)))
        is_valid, reason = is_likely_brain_mri(raw_arr)

        if not is_valid:
            return jsonify({
                'error': f'This does not appear to be a brain MRI scan. {reason} '
                         f'Please upload a valid brain MRI image.'
            }), 400

        # ── STAGE 2: run tumor classifier ────────────────────────────────────
        arr = preprocess_for_model(img, TUMOR_SIZE)
        score = predict_with_tta(tumor_model, arr, n=5)


        top_idx    = int(np.argmax(score))
        confidence = float(score[top_idx])
        prediction = CLASS_NAMES[top_idx]
        all_scores = {CLASS_NAMES[i]: round(float(score[i]) * 100, 1) for i in range(4)}

        # ── STAGE 3: confidence threshold on tumor result ─────────────────────
        if confidence < TUMOR_CONFIDENCE:
            return jsonify({
                'error': f'Image quality or scan type is unclear '
                         f'(confidence {confidence*100:.1f}%). '
                         f'Please try a clearer brain MRI scan or a different orientation.'
            }), 400

        confidence = round(confidence * 100, 1)

        labels = {
            'glioma':     'Glioma',
            'meningioma': 'Meningioma',
            'notumor':    'No Tumor',
            'pituitary':  'Pituitary Tumor',
        }
        descs = {
            'glioma':     'A malignant tumor originating in glial cells. Requires immediate specialist review.',
            'meningioma': 'A tumor arising from the meninges. Most are benign; monitoring or surgery may be advised.',
            'notumor':    'No tumor tissue detected. Brain tissue appears structurally normal.',
            'pituitary':  'A tumor at the base of the brain. Usually benign but can affect hormone levels.',
        }
        label    = labels[prediction]
        is_tumor = prediction != 'notumor'

        scan = Scan(
            user_id    = session['user_id'],
            filename   = file.filename,
            prediction = prediction,
            label      = label,
            confidence = confidence,
            is_tumor   = is_tumor,
        )
        db.session.add(scan)
        db.session.commit()

        return jsonify({
            'prediction':  prediction,
            'label':       label,
            'confidence':  confidence,
            'all_scores':  all_scores,
            'description': descs[prediction],
            'is_tumor':    is_tumor,
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Could not process image. Make sure it is a valid JPG or PNG.'}), 500

# ── Create tables ─────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
