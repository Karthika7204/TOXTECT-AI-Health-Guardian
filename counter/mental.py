# mental_health_assessment/assessment.py

def mental_health_assessment_logic(answers):
    questions = [
        "How do you feel about your overall mood today?",
        "How much energy do you have to accomplish daily tasks?",
        "How well are you sleeping these days?",
        "How would you rate your stress levels?",
        "How connected do you feel with your family and friends?",
        "How motivated are you to pursue your goals or hobbies?",
        "How often do you feel overwhelmed by daily responsibilities?",
        "How would you rate your self-esteem or self-worth?",
        "How much joy or satisfaction do you experience in your daily life?",
        "How well do you manage negative emotions like anger or sadness?"
    ]

    score = sum(answers)
    max_score = len(questions) * 5  # Maximum possible score

    # Calculate percentage score
    percentage_score = (score / max_score) * 100

    # Determine mental health state
    if percentage_score >= 80:
        mental_state = "Excellent Mental Health"
        suggestions = [
            "Keep up your healthy habits!",
            "Continue engaging in activities that bring you joy.",
            "Consider helping others—it can boost your mood even further."
        ]
    elif 60 <= percentage_score < 80:
        mental_state = "Good Mental Health"
        suggestions = [
            "You’re doing well! Maybe try adding more variety to your routine.",
            "Consider practicing mindfulness or meditation.",
            "Engage in light physical activities like yoga or walking."
        ]
    elif 40 <= percentage_score < 60:
        mental_state = "Moderate Mental Health"
        suggestions = [
            "Consider talking to a trusted friend or family member about your feelings.",
            "Spend some time in nature—it can be very uplifting.",
            "Try journaling to process your emotions and thoughts."
        ]
    else:
        mental_state = "Needs Attention"
        suggestions = [
            "Please consider seeking support from a mental health professional.",
            "Practice deep breathing or grounding techniques to reduce stress.",
            "Connect with supportive people in your life."
        ]

    return {
        'mental_state': mental_state,
        'percentage_score': percentage_score,
        'suggestions': suggestions
    }
