VARIABLES_VITALS = {
    "HR":       [220045],
    "SysABP":   [220050],
    "DiasABP":  [220051],
    "RespRate": [220210],
    "SpO2":     [220277],
    "Temp_F":   [223761],
    "Temp_C":   [223762],
}

VARIABLES_BLOOD_GAS = {
    "PaO2":             [220224],
    "SaO2_art":         [220227],
    "PaCO2":            [220235],
    "VenousPH":         [220274],
    "VenousTCO2":       [223679],
    "ArterialPH":       [223830],
    "ArterialBaseExcess":[224828],
    "ArterialTCO2":     [225698],
    "VenousPCO2":       [226062],
    "VenousPO2":        [226063],
    "AnionGap":         [227073],
    "HCO3":             [227443],
    "LacticAcid":       [225668],
}

VARIABLES_BMP = {
    "Chloride":         [220602],
    "Creatinine":       [220615],
    "Glucose_serum":    [220621],
    "Sodium_serum":     [220645],
    "BUN":              [225624],
    "Calcium":          [225625],
    "Glucose_finger":   [225664],
    "Sodium_whole":     [226534],
    "Glucose_whole":    [226537],
    "Potassium_serum":  [227442],
    "Potassium_whole":  [227464],
    "Magnesium":        [220635],
    "Phosphate":        [225677],
}

VARIABLES_CBC = {
    "Hemoglobin":   [220228],
    "HCT_serum":    [220545],
    "HCT_calc":     [226540],
    "WBC":          [220546],
    "Platelets":    [227457],
}

VARIABLES_CBC_DIFF = {
    "Basophils":    [225639],
    "Eosinophils":  [225640],
    "Lymphocytes":  [225641],
    "Monocytes":    [225642],
    "Neutrophils":  [225643],
}

VARIABLES_COAG = {
    "PT":           [227465],
    "PTT":          [227466],
    "INR":          [227467],
    "Fibrinogen":   [227468],
}

VARIABLES_LFT = {
    "AST":          [220587],
    "ALT":          [220644],
    "ALP":          [225612],
    "TotalBili":    [225690],
}

VARIABLES_CARDIAC = {
    "LDH":          [220632],
    "CK":           [225634],
    "CKMB":         [227445],
    "TroponinT":    [227429],
    "Albumin":      [227456],
}

VARIABLES_VASOPRESSORS = {
    "Epinephrine":      [221289],
    "Norepinephrine":   [221906],
    "Dopamine":         [221662],
    "Phenylephrine":    [221749],
    "Vasopressin":      [222315],
}

VARIABLES_CHARTEVENTS = {
    **VARIABLES_VITALS,
    **VARIABLES_BLOOD_GAS,
    **VARIABLES_BMP,
    **VARIABLES_CBC,
    **VARIABLES_CBC_DIFF,
    **VARIABLES_COAG,
    **VARIABLES_LFT,
    **VARIABLES_CARDIAC,
}

ALL_CHARTEVENTS_ITEMIDS = [
    iid for ids in VARIABLES_CHARTEVENTS.values() for iid in ids
]

ALL_VASOPRESSOR_ITEMIDS = [
    iid for ids in VARIABLES_VASOPRESSORS.values() for iid in ids
]

FEATURE_ORDER = (
    list(VARIABLES_VITALS.keys())
    + list(VARIABLES_BLOOD_GAS.keys())
    + list(VARIABLES_BMP.keys())
    + list(VARIABLES_CBC.keys())
    + list(VARIABLES_CBC_DIFF.keys())
    + list(VARIABLES_COAG.keys())
    + list(VARIABLES_LFT.keys())
    + list(VARIABLES_CARDIAC.keys())
    + list(VARIABLES_VASOPRESSORS.keys())
)

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_ORDER)}

ITEMID_TO_FEATURE = {}
for name, ids in {**VARIABLES_CHARTEVENTS, **VARIABLES_VASOPRESSORS}.items():
    for iid in ids:
        ITEMID_TO_FEATURE[iid] = name

N_FEATURES = len(FEATURE_ORDER)

STATIC_FEATURES = ["age", "gender", "height", "weight", "icu_type"]
N_STATIC = len(STATIC_FEATURES)

HOURLY_FEATURES = set(VARIABLES_VITALS.keys()) | set(VARIABLES_VASOPRESSORS.keys())
DAILY_FEATURES = (
    set(VARIABLES_BLOOD_GAS.keys())
    | set(VARIABLES_BMP.keys())
    | set(VARIABLES_CBC.keys())
    | set(VARIABLES_CBC_DIFF.keys())
    | set(VARIABLES_COAG.keys())
    | set(VARIABLES_LFT.keys())
    | set(VARIABLES_CARDIAC.keys())
)


if __name__ == "__main__":
    print(f"Total features: {N_FEATURES}")
    print(f"  - Vitals     : {len(VARIABLES_VITALS)}")
    print(f"  - Blood gas  : {len(VARIABLES_BLOOD_GAS)}")
    print(f"  - BMP        : {len(VARIABLES_BMP)}")
    print(f"  - CBC        : {len(VARIABLES_CBC)}")
    print(f"  - CBC diff   : {len(VARIABLES_CBC_DIFF)}")
    print(f"  - Coagulation: {len(VARIABLES_COAG)}")
    print(f"  - LFT        : {len(VARIABLES_LFT)}")
    print(f"  - Cardiac    : {len(VARIABLES_CARDIAC)}")
    print(f"  - Vasopressor: {len(VARIABLES_VASOPRESSORS)}")
    print(f"Static features: {N_STATIC}")
    print(f"CHARTEVENTS itemids: {len(ALL_CHARTEVENTS_ITEMIDS)}")
    print(f"\nFeature order (first 10): {FEATURE_ORDER[:10]}")
    print(f"Feature order (last 10): {FEATURE_ORDER[-10:]}")