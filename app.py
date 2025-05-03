from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from flask import Flask,  jsonify,render_template, request, redirect, url_for, session, flash, send_file
import random
import string
from werkzeug.utils import secure_filename
import os
import plotly.express as px
import numpy as np
import cv2
import imutils
import sklearn
from PIL import Image
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np,pandas as pd
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from google.generativeai import GenerativeModel, configure



app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# mail = Mail(app)
load_dotenv()

app.secret_key = 'MYSECRETKEY'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT') or 465)
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# mail = Mail(app)
db = SQLAlchemy(app)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(120), nullable=False)
  
    type_of_doctor = db.Column(db.String(50))

# class Appointment(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(80), nullable=False)
#     age = db.Column(db.Integer, nullable=False)
#     blood_group = db.Column(db.String(10), nullable=False)
#     time_slot = db.Column(db.String(50), nullable=False)
#     phone_number = db.Column(db.String(15), nullable=False)
#     email = db.Column(db.String(120), nullable=False)
#     type_of_doctor = db.Column(db.String(50))
#     status = db.Column(db.String(20), default='Pending')
#     prescription_file = db.Column(db.String(255))
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     user = db.relationship('User', backref=db.backref('appointments', lazy=True))

def create_tables():
    with app.app_context():
        db.create_all()

    
# ============================================================ model ============================================================ 
# Configure Mistral AI API Key

# Configure Mistral AI
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(
    model="mistral-medium",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
    streaming=False,
)

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
configure(api_key=GOOGLE_API_KEY)
gemini_model = GenerativeModel("gemini-1.5-flash")

# LangChain prompt setup for Mistral
template = """
You are SmartCare, a knowledgeable and helpful medical AI assistant. Your primary role is to:

1. Provide clear, accurate medical information in a conversational and helpful manner.
2. Always include important disclaimers when discussing medical conditions.
3. Emphasize the importance of consulting healthcare professionals.
4. Use reliable medical sources when searching for information.
5. Be empathetic and understanding when discussing health concerns.
6. For emergencies, strongly encourage seeking immediate medical attention.
7. Structure your responses clearly with relevant medical information.

Conversation History:
{history}

Human: {input}
AI:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
memory = ConversationBufferMemory(memory_key="history")
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Gemini medical image analysis prompt
sample_prompt = """You are a medical practitioner and an expert in analyzing medical-related images working for a very reputed hospital. 
You will be provided with images, and you need to identify anomalies, diseases, or health issues. 
Provide a detailed result including findings, next steps, and recommendations. 

Include a disclaimer: 'Consult with a doctor before making any decisions.' 

If certain aspects are unclear from the image, state 'Unable to determine based on the provided image.'

Now analyze the image and respond in the structured manner defined above."""

# Function to analyze image with Gemini
def analyze_image_with_gemini(image_path):
    try:
        image = Image.open(image_path)
        response = gemini_model.generate_content([sample_prompt, image])
        return response.text if response else "Error: No response received."
    except Exception as e:
        return f"Error processing image: {str(e)}"


alzheimer_model = load_model('./models/alzheimer_model.h5')
pneumonia_model = load_model('./models/pneumonia_model.h5')
skin_model = load_model('./models/skin.h5')
breastcancer_model = load_model('./models/BreastCancerModel.h5')

def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)
  
# ============================================================ routes ============================================================ 

@app.route('/', methods=['GET', 'POST'])
def index():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        
        return render_template('patient-dashboard.html', username=username)
            
    return render_template('index.html')


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        Email = user.email 
        # user_appointments = user.appointments
        return render_template('patient-profile.html', username=username,Email=Email)
        # return render_template('patient-profile.html', username=username,Email=Email, user_appointments=user_appointments)
    return render_template('index')

@app.route('/patient-register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        try:
            user = User(username=username, email=email, password=password)
            db.session.add(user)
            db.session.commit()
            session['user_id'] = user.id
            return redirect(url_for('index'))
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists. Please choose a different username.', 'error')

    return render_template('patient-register.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            flash('Wrong username or password. Please try again.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))



# ============================================================ scans ============================================================ 
    
@app.route('/breastcancer')
def breastcancer():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('breastcancer.html',username=username)
    else:
        return render_template('index.html')
@app.route('/alzheimerdetection')
def alzheimer():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('alzheimer.html',username=username)
    else:
        return render_template('index.html')
@app.route('/pneumoniadetection')
def pneumonia():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('pneumonia.html',username=username)
    else:
        return render_template('index.html')
@app.route('/skincancerdetection')
def skincancer():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('skin.html',username=username)
    else:
        return render_template('index.html')
@app.route('/resulta', methods=['GET', 'POST'])
def resulta():
    if request.method == 'POST':
        print(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (176, 176))
            img = img.reshape(1, 176, 176, 3)
            img = img/255.0
            pred = alzheimer_model.predict(img)
            pred = pred[0].argmax()
            print(pred)
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            return render_template('resulta.html', filename=filename, r=pred)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')
@app.route('/resultbc', methods=['GET', 'POST'])
def resultbc():
    if request.method == 'POST':
        print(request.url)
        
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (50, 50))
            img = np.expand_dims(img, axis=0)
            img = img/255.0
            pred = breastcancer_model.predict(img)
            predc = np.argmax(pred, axis=1).item()
            indices = {0: 'Benign', 1: 'IDC'}
            label = indices[predc]
            print(label)
           
            return render_template('resultbc.html', filename=filename, label=label)


        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/breastCancerDetection')
@app.route('/resultsc', methods=['GET', 'POST'])
def resultsc():
    if request.method == 'POST':
        print(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img,(28,28))/255
            img = np.expand_dims(img,axis=0)
            
            pred = skin_model.predict(img)
            pred = np.argmax(pred)
            labels = {
            0:'Actinic keratoses',
            1:'Basal cell carcinoma',
            2:'Seborrhoeic Keratosis',
            3:'Dermatofibroma',
            4:'Melanocytic nevi',
            5:'Vascular lesions',
            6:'Melanoma'
            
        }
            r=labels[pred]
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            return render_template('resultsc.html', filename=filename,r=r)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')


@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
    
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img/255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultp.html', filename=filename,r=pred)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get("message")  # Get text query
    image_file = request.files.get("image")  # Get image file if provided

    if not user_input and not image_file:
        return jsonify({"response": "Please provide a text message or an image for analysis."})

    responses = []

    # Process text query with Mistral if provided
    if user_input:
        mistral_response = chain.run(input=user_input).strip()
        responses.append(mistral_response)

    # Process image with Gemini if provided
    if image_file:
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        image_file.save(image_path)

        # Image display URL
        image_url = url_for("static", filename=f"uploads/{filename}", _external=True)
        gemini_response = analyze_image_with_gemini(image_path)

        responses.append(gemini_response)

    return jsonify({"response": "".join(responses)})

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response





if __name__ == '__main__':
    create_tables()
    app.run(debug=True)