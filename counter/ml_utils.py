from networkx import has_path
import easyocr
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
from django.conf import settings


csv_path = os.path.join(settings.BASE_DIR, 'counter', 'data', 'food_ingredients_and_allergens.csv')
csv_path2 = os.path.join(settings.BASE_DIR, 'counter', 'data', 'harmful_food_ingredients_expanded.csv')
df = pd.read_csv(csv_path)

ingredient_cols = ['Main Ingredient', 'Sweetener', 'Fat/Oil', 'Seasoning']
df['combined_ingredients'] = df[ingredient_cols].astype(str).agg(' '.join, axis=1)
df['label'] = LabelEncoder().fit_transform(df['Prediction'])

model = Pipeline([
    ('tfidf', TfidfVectorizer()),  
    ('clf', LogisticRegression())
])
model.fit(df['combined_ingredients'], df['label'])


harmful_df = pd.read_csv(csv_path2)
harmful_df['Ingredient'] = harmful_df['Ingredient'].str.lower().str.strip()
harmful_df['Description'] = harmful_df['Description'].str.lower().str.strip()
harmful_descriptions = {row['Ingredient']: row['Description'] for _, row in harmful_df.iterrows()}

# Add synonyms
synonyms = {
    'msg': 'monosodium glutamate',
    'bvo': 'brominated vegetable oil',
    'bht': 'butylated hydroxytoluene',
    'bha': 'butylated hydroxyanisole',
}
for k, v in synonyms.items():
    if v in harmful_descriptions:
        harmful_descriptions[k] = harmful_descriptions[v]

def image_to_text(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path, detail=0)
    cleaned_results = []
    for line in results:
        line = line.replace(';', '')
        parts = line.split(',')
        for part in parts:
            while '(' in part and ')' in part:
                open_idx = part.index('(')
                close_idx = part.index(')')
                cleaned_results.append(part[open_idx+1:close_idx].strip())
                part = part[:open_idx] + part[close_idx+1:]
            part = ' '.join([word for word in part.split() if not any(char.isdigit() for char in word)])
            cleaned_results.append(part.strip())
    return [item for item in cleaned_results if item]

def clean_ingredient(ingredient):
    return ingredient.lower().strip()

def resolve_synonym(ingredient):
    normalized = clean_ingredient(ingredient)
    return synonyms.get(normalized, normalized)

def detect_harmful_ingredients(user_input):
    resolved_input = [resolve_synonym(i) for i in user_input]
    harmful_found = {}
    for i in resolved_input:
        if i in harmful_descriptions:
            harmful_found[i] = harmful_descriptions[i]
        else:
            for key in harmful_descriptions:
                if i in key:
                    harmful_found[key] = harmful_descriptions[key]
                    break
                
    if not harmful_found:
        harmful_found["None Detected"] = "No harmful ingredients were found in the provided input."

    return harmful_found


def find_known_allergens(user_ingredients, allergen_column):
    known_allergen_set = set()
    for allergens in allergen_column.dropna():
        for item in allergens.split(','):
            known_allergen_set.add(item.strip().lower())
    detected_allergens = []
    for known_allergen in known_allergen_set:
        for user_item in user_ingredients:
            if known_allergen in user_item.lower():
                detected_allergens.append(known_allergen)
                break
    return detected_allergens

def predict_allergen_status(user_ingredients):
    allergen_hits = find_known_allergens(user_ingredients, df['Allergens'])
    if allergen_hits:
        return 'Contains'
    input_text = ' '.join(user_ingredients)
    prediction = model.predict([input_text])[0]
    return 'Contains' if prediction == 1 else 'Free'
