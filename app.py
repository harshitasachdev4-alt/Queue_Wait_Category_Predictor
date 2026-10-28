import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

st.set_page_config(
    page_title="Queue Wait Category Predictor",
    page_icon="⏳",
    layout="centered"
)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
SLOTS = [
    "09:00-10:00", "10:00-11:00", "11:00-12:00",
    "12:00-13:00", "14:00-15:00", "15:00-16:00"
]
AVG_SERVICE = 5
TIPS = {
    "Short": "You will likely be served soon.",
    "Medium": "There may be some waiting time. Check back in a while.",
    "Long": "A longer wait is expected. Consider coming later if possible."
}

def categorize(wait_minutes):
    if wait_minutes <= 15:
        return "Short"
    if wait_minutes <= 40:
        return "Medium"
    return "Long"

def generate_dataset():
    rng = np.random.default_rng(42)
    slot_factor = {"11:00-12:00": 1.3, "12:00-13:00": 1.2}
    day_factor = {"Monday": 1.2, "Friday": 1.1}
    rows = []

    for _week in range(8):
        for day in DAYS:
            for slot in SLOTS:
                n = int(rng.integers(8, 31))
                factor = slot_factor.get(slot, 1.0) * day_factor.get(day, 1.0)
                service = rng.uniform(3, 8, n) * factor
                for i in range(n):
                    rows.append({
                        "day": day,
                        "time_slot": slot,
                        "token_number": i + 1,
                        "waiting_time_min": round(float(service[:i].sum()), 1)
                    })

    df = pd.DataFrame(rows)
    df["wait_category"] = df["waiting_time_min"].apply(categorize)
    return df.drop_duplicates().reset_index(drop=True)

@st.cache_resource(show_spinner="Training the prediction models…")
def train_models():
    df = generate_dataset()
    X = df[["day", "time_slot", "token_number"]]
    y = df["wait_category"]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["day", "time_slot"]),
        ("num", StandardScaler(), ["token_number"])
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    trained, results = {}, []
    for name, estimator in models.items():
        pipeline = Pipeline([("prep", preprocess), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        predictions = pipeline.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, predictions, average="weighted", zero_division=0)
        })

    metrics = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    best_model = trained[best_name]
    report = classification_report(
        y_test, best_model.predict(X_test), zero_division=0, output_dict=True
    )
    return df, best_model, best_name, metrics, report

df, model, best_name, metrics, report = train_models()

st.title("⏳ Queue Wait Category Predictor")
st.caption("SI Lab Project | B.Tech CSE (AI & DS), Semester V | Ramdeobaba University")
st.write(
    "Enter your token number, day, and time slot to estimate the waiting category "
    "and your current position in the queue."
)

with st.form("prediction_form"):
    token_number = st.number_input(
        "Token number", min_value=1, max_value=10000, value=10, step=1
    )
    day = st.selectbox("Day", DAYS, index=0)
    time_slot = st.selectbox("Time slot", SLOTS, index=2)
    submitted = st.form_submit_button("Predict waiting time", type="primary", use_container_width=True)

if submitted:
    input_row = pd.DataFrame([{
        "day": day,
        "time_slot": time_slot,
        "token_number": int(token_number)
    }])
    category = model.predict(input_row)[0]

    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    start = datetime.strptime(time_slot.split("-")[0], "%H:%M").time()
    end = datetime.strptime(time_slot.split("-")[1], "%H:%M").time()
    served = 0
    if now.strftime("%A") == day and start <= now.time() <= end:
        elapsed = (now.hour * 60 + now.minute) - (start.hour * 60 + start.minute)
        served = int(elapsed // AVG_SERVICE)

    position = max(1, int(token_number) - served)

    st.subheader("Your prediction")
    col1, col2 = st.columns(2)
    col1.metric("Wait category", category)
    col2.metric("Estimated queue position", position)
    if category == "Short":
        st.success(TIPS[category])
    elif category == "Medium":
        st.info(TIPS[category])
    else:
        st.warning(TIPS[category])

st.divider()
st.subheader("Model performance")
st.write(f"Selected model: **{best_name}**")
display_metrics = metrics.copy()
for column in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.4f}")
st.dataframe(display_metrics, use_container_width=True, hide_index=True)

with st.expander("About the dataset"):
    st.write(
        "This project uses a simulated queue dataset, not live queue records. "
        "Waiting categories are defined as Short (up to 15 minutes), Medium "
        "(16–40 minutes), and Long (more than 40 minutes). Predictions are estimates."
    )
    st.write(f"Dataset size after duplicate removal: **{len(df):,} rows**")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

with st.expander("Classification report for the selected model"):
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.caption("Note: The queue position and waiting category are estimates based on simulated data.")
