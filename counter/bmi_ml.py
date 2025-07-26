import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from django.conf import settings
import os

# Step 1: BMI classification logic
def classify_bmi(bmi):
    if bmi < 16:
        return "Extreme Underweight"
    elif bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Extreme Overweight"

# Step 2: Weekly food plan
weekly_food_plan = {
    "Extreme Underweight": {
        "Day 1": "High-calorie smoothies, nuts, eggs, whole milk",
        "Day 2": "Peanut butter toast, oatmeal with cream",
        "Day 3": "Cheese omelet, avocado toast",
        "Day 4": "Greek yogurt parfait with honey",
        "Day 5": "Grilled chicken and brown rice",
        "Day 6": "Pasta with creamy sauce",
        "Day 7": "Steak with mashed potatoes"
    },
    "Underweight": {
        "Day 1": "Egg sandwich, fruit juice",
        "Day 2": "Granola with yogurt",
        "Day 3": "Veggie wrap with hummus",
        "Day 4": "Chicken soup and bread",
        "Day 5": "Rice and beans",
        "Day 6": "Baked salmon with veggies",
        "Day 7": "Almond butter on toast"
    },
    "Normal": {
        "Day 1": "Balanced smoothie with spinach and banana",
        "Day 2": "Eggs and toast",
        "Day 3": "Grilled chicken salad",
        "Day 4": "Whole grain pasta",
        "Day 5": "Fish with steamed vegetables",
        "Day 6": "Rice bowl with veggies",
        "Day 7": "Lentil soup with whole wheat bread"
    },
    "Overweight": {
        "Day 1": "Fruit bowl, boiled egg",
        "Day 2": "Oats with skim milk",
        "Day 3": "Vegetable salad with lemon",
        "Day 4": "Grilled fish and quinoa",
        "Day 5": "Steamed chicken with greens",
        "Day 6": "Soup and salad",
        "Day 7": "Smoothie with flaxseeds"
    },
    "Extreme Overweight": {
        "Day 1": "Green smoothie, no sugar",
        "Day 2": "Oatmeal with berries",
        "Day 3": "Steamed veggies and tofu",
        "Day 4": "Grilled chicken breast with kale",
        "Day 5": "Low-carb salad",
        "Day 6": "Broccoli and cauliflower stir fry",
        "Day 7": "Herbal tea and fruit snack"
    }
}

# Step 3: Load dataset
path = os.path.join(settings.BASE_DIR, 'counter', 'data', 'bmi_dataset.csv')
df = pd.read_csv(path)

# Step 4: Train model
X = df[["Height", "Weight"]]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Prediction function for Django view
def get_bmi_prediction_and_plan(height_cm, weight_kg):
    height_m = height_cm / 100  # Convert to meters
    prediction = model.predict([[height_m, weight_kg]])[0]
    plan = weekly_food_plan.get(prediction, {})
    return prediction, plan

