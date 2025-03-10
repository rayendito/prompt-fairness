import pandas as pd
PROFILE_COLUMNS = [
    "Employment-sector",
    "Ethnicity simplified",
    "Highest education level completed",
    "Sex",
    "Age",
    "English speaking monolingual",
]

def get_empty_question_profile(empty=[]):
    return {
        "employment_sector" : {
            "Health-STEM" : empty.copy(),
            "Engineering-STEM" : empty.copy(),
            "Humanities" : empty.copy(),
        },
        "race" : {
            "White" : empty.copy(),
            "Asian" : empty.copy(),
            "Mixed" : empty.copy(),
            "Black" : empty.copy(),
            "Other" : empty.copy(),
        },
        "education" : {
            "Highschool" : empty.copy(),
            "Graduate" : empty.copy(),
            "Postgraduate" : empty.copy(),
        },
        "sex" : {
            "male" : empty.copy(),
            "female" : empty.copy(),
        },
        "age" : {
            "18-30" : empty.copy(),
            "31-40" : empty.copy(),
            "41-50" : empty.copy(),
        },
        "language" : {
            "L1" : empty.copy(),
            "L2" : empty.copy(),
        },
    }

def update_question_profile(empty_profile, row, filler):
    if(filler == "population"):
        value = [1]
    else:
        value = [row[filler]]
    
    for pc in PROFILE_COLUMNS:
        if (pc == "Employment-sector"):
            if row[pc] in ["Medicine"]:
                empty_profile["employment_sector"]["Health-STEM"] += value
            elif row[pc] in ["Manufacturing", "Science, Technology, Engineering & Mathematics", "Technology and Telecommunications", "Manufacturing and Industry"]:
                empty_profile["employment_sector"]["Engineering-STEM"] += value
            elif row[pc] in ["Arts", "Hospitality & Tourism", "Education & Training", "Policing", "Government & Public Administration", "Business Management & Administration", "Legal", "Education and Research", "Hospitality and Tourism"]:
                empty_profile["employment_sector"]["Humanities"] += value
        elif (pc == "Ethnicity simplified"):
            if row[pc] == "White" :
                empty_profile["race"]["White"] += value
            elif row[pc] == "Asian":
                empty_profile["race"]["Asian"] += value
            elif row[pc] == "Mixed":
                empty_profile["race"]["Mixed"] += value
            elif row[pc] == "Black":
                empty_profile["race"]["Black"] += value
            elif row[pc] == "Other":
                empty_profile["race"]["Other"] += value
        elif (pc == "Highest education level completed"):
            if row[pc] in ["High school diploma/A-levels", "Technical/community college", "Highschool"]:
                empty_profile["education"]["Highschool"] += value
            elif row[pc] in ["Undergraduate degree (BA/BSc/other)", "Bachelor degree"]:
                empty_profile["education"]["Graduate"] += value
            elif row[pc] in ["Graduate degree (MA/MSc/MPhil/other)", "Doctorate degree (PhD/other)", "Master degree"]:
                empty_profile["education"]["Postgraduate"] += value
        elif (pc == "Sex"):
            if row[pc] == "Male":
                empty_profile["sex"]["male"] += value
            else:
                empty_profile["sex"]["female"] += value
        elif (pc == "Age"):
            if 18 <= row[pc] and row[pc] <= 30 :
                empty_profile["age"]["18-30"] += value
            elif 31 <= row[pc] and row[pc] <= 40 :
                empty_profile["age"]["31-40"] += value
            elif 41 <= row[pc] and row[pc] <= 50 :
                empty_profile["age"]["41-50"] += value
        elif (pc == "English speaking monolingual"):
            if row[pc] in ["I only know English"]:
                empty_profile["language"]["L1"] += value
            elif row[pc] in ["Guaranteed non English L1"]:
                empty_profile["language"]["L2"] += value

def reduce_dict(d, method="avg"):
    def compute(lst):
        print(lst)
        cleaned = [x for x in lst if not (isinstance(x, float) and pd.isna(x))]
        print(cleaned)
        if not cleaned:
            return 0
        return sum(cleaned) if method == "sum" else sum(cleaned) / len(cleaned)

    if isinstance(d, dict):
        return {k: reduce_dict(v, method) for k, v in d.items()}
    elif isinstance(d, list):
        return compute(d)
    else:
        return d