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

# ── SQLite config ─────────────────────────────────────────────────────────────
# Creates a file called neuroscan.db in your project folder automatically
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

    def __repr__(self):
        return f'<User {self.email}>'

class Scan(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename    = db.Column(db.String(256))
    prediction  = db.Column(db.String(50))   # e.g. 'glioma'
    label       = db.Column(db.String(100))  # e.g. 'Glioma'
    confidence  = db.Column(db.Float)        # e.g. 94.3
    is_tumor    = db.Column(db.Boolean)
    scanned_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Scan {self.prediction} {self.confidence}%>'

# ── Load ML model ─────────────────────────────────────────────────────────────
model = load_model(r"C:\Users\ujjawal baliyan\Downloads\brain_tumor_model2.keras")
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 180

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
    # Last 5 scans for the recent history table
    scans = (Scan.query.filter_by(user_id=user.id)
                 .order_by(Scan.scanned_at.desc())
                 .limit(5).all())
    # Quick stats
    total      = Scan.query.filter_by(user_id=user.id).count()
    tumor_ct   = Scan.query.filter_by(user_id=user.id, is_tumor=True).count()
    no_tumor   = Scan.query.filter_by(user_id=user.id, is_tumor=False).count()
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
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(np.array(img), axis=0)

       # raw   = model.predict(arr, verbose=0)
        score = model.predict(arr,verbose=0)[0]

        top_idx    = int(np.argmax(score))
        prediction = CLASS_NAMES[top_idx]
        confidence = round(float(score[top_idx]) * 100, 1)
        all_scores = {CLASS_NAMES[i]: round(float(score[i]) * 100, 1) for i in range(4)}

        labels = {
            'glioma':     'Glioma',
            'meningioma': 'Meningioma',
            'notumor':    'No Tumor',
            'pituitary':  'Pituitary Tumor',
        }
        descs = {
            'glioma':     'A malignant tumor originating in glial cells of the brain or spinal cord. Requires immediate specialist review.',
            'meningioma': 'A tumor arising from the meninges. Most are benign and slow-growing; monitoring or surgery may be advised.',
            'notumor':    'No tumor tissue detected. Brain tissue appears structurally normal.',
            'pituitary':  'A tumor at the base of the brain in the pituitary gland. Usually benign but can affect hormone levels.',
        }
        label    = labels[prediction]
        is_tumor = prediction != 'notumor'

        # ── Save scan to database ──────────────────────────────────────────────
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

# ── Create tables on first run ────────────────────────────────────────────────
with app.app_context():
    db.create_all()   # Safe to call repeatedly — only creates tables if missing

if __name__ == '__main__':
    app.run(debug=True)