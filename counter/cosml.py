import os
import easyocr
import pandas as pd
from django.conf import settings


# Path to the harmful chemicals dataset
DATASET_PATH = os.path.join(settings.BASE_DIR, 'counter', 'data', 'cscpopendata.csv')


def images_to_text(image_path):
    # Initialize the EasyOCR Reader
    reader = easyocr.Reader(['en'])

    # Perform text detection on the image
    results = reader.readtext(image_path, detail=0)

    # Clean and format the detected text
    cleaned_results = []
    for line in results:
        line_without_semicolon = line.replace(';', '')
        parts = line_without_semicolon.split(',')
        for part in parts:
            while '(' in part and ')' in part:
                open_idx = part.index('(')
                close_idx = part.index(')')
                # Extract content within parentheses and add it as a separate item
                cleaned_results.append(part[open_idx + 1:close_idx].strip())
                part = part[:open_idx] + part[close_idx + 1:]
            # Remove numerical percentages
            part = ' '.join([word for word in part.split() if not any(char.isdigit() for char in word)])
            cleaned_results.append(part.strip())

    # Filter out empty strings
    return [item for item in cleaned_results if item]


def detect_harmful_chemicals(user_ingredients):
    """
    Detect harmful chemicals by comparing user ingredients with a dataset.

    Args:
        user_ingredients (list): List of ingredients provided by the user.

    Returns:
        list: List of harmful ingredients found in the user input.
    """
    try:
        # Load the dataset
        dataset = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        raise Exception("Dataset file not found. Please ensure the file exists at the specified path.")

    # Ensure the 'ChemicalName' column exists
    if 'ChemicalName' not in dataset.columns:
        raise Exception("The dataset must contain a 'ChemicalName' column.")

    # Convert the 'ChemicalName' column to a set for faster lookup
    harmful_chemicals = set(dataset['ChemicalName'].str.lower())

    # Find matches
    matched_chemicals = [ingredient for ingredient in user_ingredients if ingredient.lower() in harmful_chemicals]

    return matched_chemicals


