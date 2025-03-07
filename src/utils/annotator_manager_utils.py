PROFILE_COLUMNS = [
    "Employment-sector",
    "Ethnicity simplified",
    "Highest education level completed",
    "Sex",
    "Age",
    "English speaking monolingual",
]

def get_empty_question_profile(empty_value=0):
    return {
        "employment_sector" : {
            "Health-STEM" : empty_value,
            "Engineering-STEM" : empty_value,
            "Humanities" : empty_value,
        },
        "race" : {
            "White" : empty_value,
            "Asian" : empty_value,
            "Mixed" : empty_value,
            "Black" : empty_value,
            "Other" : empty_value,
        },
        "education" : {
            "Highschool" : empty_value,
            "Graduate" : empty_value,
            "Postgraduate" : empty_value,
        },
        "sex" : {
            "male" : empty_value,
            "female" : empty_value,
        },
        "age" : {
            "18-30" : empty_value,
            "31-40" : empty_value,
            "41-50" : empty_value,
        },
        "language" : {
            "L1" : empty_value,
            "L2" : empty_value,
        },
    }

def update_question_profile(empty_profile, row, filler="population"):
    if(filler == "population"):
        value = 1
    else:
        value = [row[filler]]
    
    for pc in PROFILE_COLUMNS:
        if (pc == "Employment-sector"):
            if row[pc] in ["Medicine"]:
                empty_profile["employment_sector"]["Health-STEM"] += 1
            elif row[pc] in ["Manufacturing", "Science, Technology, Engineering & Mathematics", "Technology and Telecommunications", "Manufacturing and Industry"]:
                empty_profile["employment_sector"]["Engineering-STEM"] += 1
            elif row[pc] in ["Arts", "Hospitality & Tourism", "Education & Training", "Policing", "Government & Public Administration", "Business Management & Administration", "Legal", "Education and Research", "Hospitality and Tourism"]:
                empty_profile["employment_sector"]["Humanities"] += 1
        elif (pc == "Ethnicity simplified"):
            if row[pc] == "White" :
                empty_profile["race"]["White"] += 1
            elif row[pc] == "Asian":
                empty_profile["race"]["Asian"] += 1
            elif row[pc] == "Mixed":
                empty_profile["race"]["Mixed"] += 1
            elif row[pc] == "Black":
                empty_profile["race"]["Black"] += 1
            elif row[pc] == "Other":
                empty_profile["race"]["Other"] += 1
        elif (pc == "Highest education level completed"):
            if row[pc] in ["High school diploma/A-levels", "Technical/community college", "Highschool"]:
                empty_profile["education"]["Highschool"] += 1
            elif row[pc] in ["Undergraduate degree (BA/BSc/other)", "Bachelor degree"]:
                empty_profile["education"]["Graduate"] += 1
            elif row[pc] in ["Graduate degree (MA/MSc/MPhil/other)", "Doctorate degree (PhD/other)", "Master degree"]:
                empty_profile["education"]["Postgraduate"] += 1
        elif (pc == "Sex"):
            if row[pc] == "Male":
                empty_profile["sex"]["male"] += 1
            else:
                empty_profile["sex"]["female"] += 1
        elif (pc == "Age"):
            if 18 <= row[pc] and row[pc] <= 30 :
                empty_profile["age"]["18-30"] += 1
            elif 31 <= row[pc] and row[pc] <= 40 :
                empty_profile["age"]["31-40"] += 1
            elif 41 <= row[pc] and row[pc] <= 50 :
                empty_profile["age"]["41-50"] += 1
        elif (pc == "English speaking monolingual"):
            if row[pc] in ["I only know English"]:
                empty_profile["language"]["L1"] += 1
            elif row[pc] in ["Guaranteed non English L1"]:
                empty_profile["language"]["L2"] += 1
