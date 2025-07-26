from django.shortcuts import render, redirect
import requests
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User, auth
from django.contrib import messages
from PIL import Image
from .ml_utils import image_to_text, predict_allergen_status, find_known_allergens, detect_harmful_ingredients
from django.core.files.storage import FileSystemStorage
import os
import pandas as pd
from django.conf import settings
from .cosml import images_to_text, detect_harmful_chemicals
import numpy as np
import pickle
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .bmi_ml import get_bmi_prediction_and_plan
import json
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Tablet
from datetime import datetime
from django.core.mail import send_mail
from django.utils import timezone
from .models import Tablet
from .mental import mental_health_assessment_logic
from django.shortcuts import render
import joblib
import numpy as np



def home(request):
    return render(request, 'home.html')

def auth_page(request):
    return render(request,"auth.html")


def user_signup(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('auth_page')

        if User.objects.filter(username=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('auth_page')

        user = User.objects.create_user(username=email, email=email, password=password)
        login(request, user)
        return redirect('dashboard')  

    return redirect('auth_page')

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(request, username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('dashboard')  
        else:
            messages.info(request, 'Invalid credentials')
            return redirect('auth_page')

    return redirect('auth_page')

def user_logout(request):
    logout(request)
    return redirect('auth-page')


def dashboard(request):
    return render(request, 'dashboard.html')

def scan(request):
    csv_path = os.path.join(settings.BASE_DIR, 'counter', 'data', 'food_ingredients_and_allergens.csv')
    df = pd.read_csv(csv_path)
    context = {}
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        image_path = fs.path(filename)

        with open(image_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        extracted_text = image_to_text(image_path)
        cleaned_input = [word.lower().strip() for word in extracted_text]

        allergen_status = predict_allergen_status(cleaned_input)
        allergens = find_known_allergens(cleaned_input, df['Allergens'])
        harmful = detect_harmful_ingredients(cleaned_input)

        

        context = {
            'status': allergen_status,
            'allergens': allergens,
            'harmful': harmful,
            'text': extracted_text
        }

    return render(request, 'scan.html', context)

def process_image(request):
    """
    Django view to handle image upload, text extraction, and rendering the output in HTML.
    """
    if request.method == 'POST' and 'image' in request.FILES:
        # Save the uploaded image to the MEDIA_ROOT directory
        uploaded_image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)

        # Write the uploaded image to disk
        with open(image_path, 'wb') as f:
            for chunk in uploaded_image.chunks():
                f.write(chunk)

        # Extract text from the image
        extracted_text = images_to_text(image_path)

        # Check for harmful chemicals
        dataset_path = os.path.join(settings.BASE_DIR, 'counter', 'data', 'cscpopendata.csv')
        harmful_chemicals = detect_harmful_chemicals(extracted_text)

        # Render the output
        return render(request, 'cosm.html', {
            "status": "success",
            "extracted_text": extracted_text,
            "harmful_chemicals": harmful_chemicals,
        })

    # Handle invalid request
    return render(request, 'cosm.html')


def calories(request):
    api = None
    if request.method == 'POST':
        query = request.POST.get('query')
        api_url = 'https://api.calorieninjas.com/v1/nutrition?query='
        headers = {
            'X-Api-Key': 'J4GM9+IvPcPEU3vt5eS9YA==NBvKuVWyLQapw9GV'
        }
        try:
            api_request = requests.get(api_url + query, headers=headers)
            api_request.raise_for_status()  # Raise an error for bad status codes
            api = api_request.json()  # Parse JSON response
            print("API Response:", api)  # Debug print
        except requests.exceptions.RequestException as e:
            api = f"Oops! There was an error: {e}"
            print(e)
    
    return render(request, 'calories.html', {'api': api})


# Load datasets
# Updated CSV loading using settings.BASE_DIR
sym_des = pd.read_csv(os.path.join(settings.BASE_DIR, 'counter', 'data', 'symtoms_df.csv'))
precautions = pd.read_csv(os.path.join(settings.BASE_DIR, 'counter', 'data', 'precautions_df.csv'))
workout = pd.read_csv(os.path.join(settings.BASE_DIR, 'counter', 'data', 'workout_df.csv'))
description = pd.read_csv(os.path.join(settings.BASE_DIR, 'counter', 'data', 'description.csv'))
medications = pd.read_csv(os.path.join(settings.BASE_DIR, 'counter', 'data', 'medications.csv'))
diets = pd.read_csv(os.path.join(settings.BASE_DIR, 'counter', 'data', 'diets.csv'))

# Load model
svc = pickle.load(open(r'E:\Foodie-calorie-finder-main\counter\model\svc.pkl', 'rb'))

# Symptoms and disease dictionary
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Predict Function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Index and prediction view
@csrf_exempt
def health(request):
    symptoms = None
    if request.method == "POST":
        symptoms = request.POST.get("symptoms")

        if symptoms == "Symptoms" or not symptoms:
            return render(request, "index.html", {"message": "Please either write symptoms or you have written misspelled symptoms"})

        user_symptoms = [s.strip("[]' ") for s in symptoms.split(',')]

        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions_, medications_, rec_diet, workout_ = helper(predicted_disease)

        my_precautions = []
        for i in precautions_[0]:
            my_precautions.append(i)

        return render(request, "health.html", {
            "predicted_disease": predicted_disease,
            "dis_des": dis_des,
            "my_precautions": my_precautions,
            "medications": medications_,
            "my_diet": rec_diet,
            "workout": workout_,
        })

    return render(request, "health.html")

def bmi_predictor(request):
    prediction = None
    food_plan = None

    if request.method == 'POST':
        height = float(request.POST.get('height'))
        weight = float(request.POST.get('weight'))
        prediction, food_plan = get_bmi_prediction_and_plan(height, weight)

    return render(request, 'bmi.html', {
        'prediction': prediction,
        'food_plan': food_plan,
    })

def vision_check(request):
    return render(request, 'visioncheck.html')

def tablet_tracker(request):
    return render(request,'tablet.html')

@login_required
def save_tablets(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        tablets = data.get('tablets', [])

        user_email = request.user.email if request.user.is_authenticated else None

        if not user_email:
            return JsonResponse({'error': 'User email not found'}, status=400)

        for tab in tablets:
            tablet_obj = Tablet.objects.create(
                user=request.user,
                name=tab['name'],
                count=tab['count'],
                dosage=tab['dosage'],
                start_date=datetime.strptime(tab['startDate'], '%Y-%m-%d').date(),
                finish_date=datetime.strptime(tab['finishDate'], '%Y-%m-%d').date()
            )

            # Calculate days left
            today = timezone.now().date()
            finish_date = tablet_obj.finish_date
            days_left = (finish_date - today).days

            # Check if the tablet is near to finishing
            if 0 < days_left <= 3:
                # Send email
                send_mail(
                    subject="Tablet Finish Reminder",
                    message=f"Reminder: Your tablet '{tablet_obj.name}' will finish in {days_left} days.",
                    from_email='gayathrirajendran3967@gmail.com',  # Replace this
                    recipient_list=[user_email],
                    fail_silently=False,
                )

        return JsonResponse({'message': 'Tablets saved successfully!'})

    return JsonResponse({'error': 'Invalid request'}, status=400)

def start_assessment(request):
    if request.method == "POST":
        answers = request.POST.getlist('answers')
        answers = [int(answer) for answer in answers]

        result = mental_health_assessment_logic(answers)

        return JsonResponse(result)

    return render(request, 'mentalhealth.html')

model = pickle.load(open(r'E:\Foodie-calorie-finder-main\counter\model\heart_attack_model.pkl', 'rb'))
 

def predict_heart_attack(request):
    if request.method == 'POST':
        try:
            age = int(request.POST.get('age'))
            gender_input = request.POST.get('gender')
            gender = 1 if gender_input.lower() == 'male' else 0
            diabetes = int(request.POST.get('diabetes'))
            hypertension = int(request.POST.get('hypertension'))
            obesity = int(request.POST.get('obesity'))
            smoking = int(request.POST.get('smoking'))
            alcohol = int(request.POST.get('alcohol'))
            physical_activity = int(request.POST.get('physical_activity'))
            stress = int(request.POST.get('stress'))
            healthcare_access = int(request.POST.get('healthcare_access'))
            heart_attack_history = int(request.POST.get('heart_attack_history'))

            features = np.array([[age, gender, diabetes, hypertension, obesity, 
                                  smoking, alcohol, physical_activity, stress,
                                  healthcare_access, heart_attack_history]])
            prediction = model.predict(features)[0]

            if prediction == 1:
                risk = "High risk of heart attack"
                suggestions = [
                    "Quit smoking and limit alcohol consumption.",
                    "Engage in regular physical activity.",
                    "Maintain a balanced diet rich in fruits and vegetables.",
                    "Manage stress effectively.",
                    "Attend regular health check-ups."
                ]
            else:
                risk = "Low risk of heart attack"
                suggestions = [
                    "Continue maintaining a healthy lifestyle.",
                    "Engage in regular exercise and balanced diet.",
                    "Avoid smoking and limit alcohol intake.",
                    "Keep stress levels under control.",
                    "Stay updated with regular health screenings."
                ]

            return render(request, 'heart.html', {
                'prediction': risk,
                'suggestions': suggestions
            })

        except Exception as e:
            print(e)
            return render(request, 'heart.html', {'prediction': 'Invalid input', 'suggestions': []})
    else:
        return render(request, 'heart.html')