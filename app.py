from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_babel import Babel, _
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-super-secret-key-123456789-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root@localhost/waste_classifier'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Languages
LANGUAGES = {
    'en': 'English',
    'hi': 'हिंदी',
    'bn': 'বাংলা',
    'es': 'Español',
    'fr': 'Français'
}

def get_locale():
    lang = request.args.get('lang')
    if lang in LANGUAGES:
        session['lang'] = lang
        return lang
    if 'lang' in session:
        return session['lang']
    return request.accept_languages.best_match(LANGUAGES.keys())

babel = Babel(app, locale_selector=get_locale)

@app.context_processor
def inject_locale():
    return dict(get_locale=get_locale)

@app.context_processor
def inject_languages():
    return dict(LANGUAGES=LANGUAGES)

@app.route('/change-language/<lang>')
def change_language(lang):
    if lang in LANGUAGES:
        session['lang'] = lang
    return redirect(request.referrer or url_for('landing'))

# ==================== DATABASE ====================
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()

# ==================== MODEL ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load('resnet50_waste_finetuned.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

recyclability = {
    'cardboard': _('Recyclable'),
    'glass': _('Recyclable'),
    'metal': _('Recyclable'),
    'paper': _('Recyclable'),
    'plastic': _('Recyclable'),
    'trash': _('Non-recyclable')
}

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, idx = torch.max(probs, 0)
        return class_names[idx.item()], confidence.item() * 100

# ==================== DECORATORS ====================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash(_('Please log in first.'), 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash(_('No file selected'), 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash(_('No file selected'), 'error')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Prevent duplicates (last 5 minutes)
        five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
        recent_pred = Prediction.query.filter(
            Prediction.user_id == session['user_id'],
            Prediction.image_name == filename,
            Prediction.timestamp > five_minutes_ago
        ).first()

        if recent_pred:
            prediction = recent_pred.prediction
            confidence = recent_pred.confidence
            flash(_('Same image detected — showing previous result'), 'info')
        else:
            prediction, confidence = predict_image(filepath)
            new_pred = Prediction(
                image_name=filename,
                prediction=prediction,
                confidence=confidence,
                user_id=session['user_id']
            )
            db.session.add(new_pred)
            db.session.commit()

        return render_template('classification_result.html',
                               prediction=prediction,
                               confidence=f"{confidence:.1f}",
                               image_url=url_for('static', filename=f'uploads/{filename}'),
                               recyclability=recyclability)

    return render_template('upload_image.html')

@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=session['user_id'])\
        .order_by(Prediction.timestamp.desc()).all()
    return render_template('prediction_history.html', history=predictions)

@app.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    return render_template('user_profile.html', user=user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash(_('Username already taken'), 'error')
        elif User.query.filter_by(email=email).first():
            flash(_('Email already registered'), 'error')
        else:
            new_user = User(username=username, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash(_('Registration successful! Please log in.'), 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.password == password:
            session['user_id'] = user.id
            flash(_('Welcome back, {name}!').format(name=user.username), 'success')
            return redirect(url_for('upload_image'))
        else:
            flash(_('Invalid username or password'), 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash(_('Logged out successfully'), 'success')
    return redirect(url_for('landing'))

if __name__ == '__main__':
    app.run(debug=True)