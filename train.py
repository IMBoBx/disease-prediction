import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading datasets...")
df = pd.read_csv("./data/dataset.csv")
severity_df = pd.read_csv("./data/Symptom-severity.csv")

# Strip whitespace from column names and values
df.columns = df.columns.str.strip()
severity_df.columns = severity_df.columns.str.strip()
severity_df['Symptom'] = severity_df['Symptom'].str.strip().str.lower().str.replace(' ', '_')

# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
print("Preprocessing...")

# Get all unique symptoms
symptom_cols = [c for c in df.columns if c != 'Disease']
all_symptoms = set()
for col in symptom_cols:
    df[col] = df[col].astype(str).str.strip().str.lower().str.replace(' ', '_')
    all_symptoms.update(df[col].unique())
all_symptoms = {s for s in all_symptoms if isinstance(s, str) and s != 'nan'}
all_symptoms = sorted(list(all_symptoms))

# Binary encode: one column per symptom
def encode_row(row):
    present = set(row[symptom_cols].values)
    return {s: 1 if s in present else 0 for s in all_symptoms}

print(f"Found {len(all_symptoms)} unique symptoms across {len(df)} samples.")

encoded = df.apply(encode_row, axis=1, result_type='expand')
encoded['Disease'] = df['Disease'].str.strip()

X = encoded[all_symptoms]
y = encoded['Disease']

# ── Noise injection ──────────────────────────
# The dataset is deterministic (fixed symptom sets per disease),
# so we inject noise to simulate real-world incomplete reporting.
print("Injecting noise for realistic evaluation...")
rng = np.random.default_rng(42)

X_noisy = X.copy().astype(float)

# 1. Randomly drop present symptoms (simulate patient not reporting them)
drop_mask = (X_noisy == 1) & (rng.random(X_noisy.shape) < 0.15)
X_noisy[drop_mask] = 0

# 2. Randomly add spurious symptoms (simulate co-morbidities / noise)
add_mask = (X_noisy == 0) & (rng.random(X_noisy.shape) < 0.05)
X_noisy[add_mask] = 1

X = X_noisy

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# ─────────────────────────────────────────────
# 3. SEVERITY REGRESSION
# ─────────────────────────────────────────────
print("\nTraining severity regression model...")

severity_map = dict(zip(severity_df['Symptom'], severity_df['weight']))

def compute_severity(row):
    total = 0
    for s in all_symptoms:
        if row[s] == 1:
            total += severity_map.get(s, 0)
    return total

X_sev = X.copy()
y_sev = X_sev.apply(compute_severity, axis=1)

X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
    X_sev, y_sev, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_sev, y_train_sev)
y_pred_sev = reg_model.predict(X_test_sev)

print(f"  Severity Regression → MSE: {mean_squared_error(y_test_sev, y_pred_sev):.4f} | R²: {r2_score(y_test_sev, y_pred_sev):.4f}")

# ─────────────────────────────────────────────
# 4. CLASSIFICATION MODEL COMPARISON
# ─────────────────────────────────────────────
print("\nTraining and comparing classifiers...\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {"accuracy": round(acc * 100, 2), "f1": round(f1, 4)}
    trained_models[name] = model
    print(f"  {name:<22} → Accuracy: {acc*100:.2f}%  |  F1: {f1:.4f}")

# ─────────────────────────────────────────────
# 5. SAVE BEST MODEL
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['accuracy'])
print(f"\nBest model: {best_name} ({results[best_name]['accuracy']}% accuracy)")

joblib.dump(trained_models[best_name], "disease_model.pkl")
joblib.dump(reg_model, "severity_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(all_symptoms, "symptoms_list.pkl")

# Save metadata for the frontend and results display
with open("model_meta.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "comparison": results,
        "num_symptoms": len(all_symptoms),
        "num_diseases": len(le.classes_),
        "diseases": list(le.classes_)
    }, f, indent=2)

print("\nSaved: disease_model.pkl, severity_model.pkl, label_encoder.pkl, symptoms_list.pkl, model_meta.json")
print("\n─── Full Classification Report (Best Model) ───")
y_pred_best = trained_models[best_name].predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=le.classes_))