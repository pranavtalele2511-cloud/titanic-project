from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
MODEL_PATH = BASE_DIR / "all_models.pkl"
FEATURE_COLUMNS = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S",
]
MODEL_NAMES = {
    "log": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Machine",
}


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    saved_models = joblib.load(MODEL_PATH)
    return {
        str(label).strip().lower(): (
            model,
            MODEL_NAMES.get(
                str(label).strip().lower(), str(label).strip().upper()
            ),
        )
        for model, label in saved_models.items()
    }


def prepare_features(data):
    features = data.copy()
    features["Age"] = features["Age"].fillna(features["Age"].mean())
    features["Embarked"] = features["Embarked"].fillna(
        features["Embarked"].mode().iloc[0]
    )
    features["FamilySize"] = features["SibSp"] + features["Parch"] + 1
    features["IsAlone"] = (features["FamilySize"] == 1).astype(int)
    features = pd.get_dummies(
        features[["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize",
                  "IsAlone", "Sex", "Embarked"]],
        columns=["Sex", "Embarked"],
        drop_first=True,
    )
    return features.reindex(columns=FEATURE_COLUMNS, fill_value=0)


def make_passenger_features(pclass, sex, age, sibsp, parch, fare, embarked):
    family_size = sibsp + parch + 1
    return pd.DataFrame(
        [[
            pclass,
            age,
            sibsp,
            parch,
            fare,
            family_size,
            int(family_size == 1),
            int(sex == "Male"),
            int(embarked == "Q"),
            int(embarked == "S"),
        ]],
        columns=FEATURE_COLUMNS,
    )


st.set_page_config(page_title="Titanic Model Explorer", page_icon="🚢", layout="wide")
st.title("🚢 Titanic Survival Model Explorer")
st.caption("Explore the passenger data, compare the saved classifiers, and try a prediction.")

try:
    passengers = load_data()
    models = load_models()
except FileNotFoundError as error:
    st.error(f"Required project file not found: {error.filename}")
    st.stop()

if not models:
    st.error("No saved models were found in all_models.pkl.")
    st.stop()

feature_data = prepare_features(passengers)
labels = passengers["Survived"]
_, X_test, _, y_test = train_test_split(
    feature_data, labels, test_size=0.20, random_state=42
)

performance = []
for key, (model, display_name) in models.items():
    predictions = model.predict(X_test)
    performance.append(
        {
            "Model": display_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "model_key": key,
        }
    )
performance_df = pd.DataFrame(performance)

sidebar_model = st.sidebar.selectbox(
    "Choose a model",
    options=list(models),
    format_func=lambda key: models[key][1],
)
selected_model, selected_model_name = models[sidebar_model]

overview_tab, prediction_tab, data_tab = st.tabs(
    ["Model overview", "Try a prediction", "Passenger data"]
)

with overview_tab:
    survived_count = int(passengers["Survived"].sum())
    survival_rate = passengers["Survived"].mean() * 100
    metric_columns = st.columns(4)
    metric_columns[0].metric("Passengers", f"{len(passengers):,}")
    metric_columns[1].metric("Survived", f"{survived_count:,}")
    metric_columns[2].metric("Survival rate", f"{survival_rate:.1f}%")
    selected_accuracy = performance_df.loc[
        performance_df["model_key"] == sidebar_model, "Accuracy"
    ].iloc[0]
    metric_columns[3].metric("Selected model accuracy", f"{selected_accuracy:.1%}")

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.subheader("Survival outcome")
        outcome_counts = (
            passengers["Survived"]
            .map({0: "Did not survive", 1: "Survived"})
            .value_counts()
            .rename_axis("Outcome")
            .to_frame("Passengers")
        )
        st.bar_chart(outcome_counts)
    with chart_columns[1]:
        st.subheader("Survival rate by passenger class")
        class_rates = (
            passengers.groupby("Pclass")["Survived"]
            .mean()
            .mul(100)
            .rename("Survival rate (%)")
            .to_frame()
        )
        st.bar_chart(class_rates)

    st.subheader("Saved model comparison")
    st.bar_chart(performance_df.set_index("Model")[["Accuracy"]])
    st.caption("Accuracy is measured on a held-out 20% split with random_state=42.")

    if hasattr(selected_model, "coef_"):
        st.subheader("Logistic regression feature weights")
        weights = pd.Series(
            selected_model.coef_[0], index=FEATURE_COLUMNS, name="Weight"
        ).sort_values()
        st.bar_chart(weights)
        st.caption("Positive weights favor survival; negative weights favor non-survival.")

    st.subheader(f"{selected_model_name} confusion matrix")
    selected_predictions = selected_model.predict(X_test)
    matrix = confusion_matrix(y_test, selected_predictions, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual: did not survive", "Actual: survived"],
            columns=["Predicted: did not survive", "Predicted: survived"],
        ),
        width="stretch",
    )

with prediction_tab:
    st.subheader("Enter passenger details")
    left, right = st.columns(2)
    with left:
        pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["Female", "Male"])
        age = st.slider("Age", min_value=0, max_value=100, value=30)
        fare = st.number_input("Fare", min_value=0.0, value=32.20, step=1.0)
    with right:
        sibsp = st.number_input(
            "Siblings / spouses aboard", min_value=0, max_value=10, value=0
        )
        parch = st.number_input(
            "Parents / children aboard", min_value=0, max_value=10, value=0
        )
        embarked = st.selectbox("Port of embarkation", ["C", "Q", "S"], index=2)

    if st.button("Predict survival", type="primary"):
        passenger = make_passenger_features(
            pclass, sex, age, sibsp, parch, fare, embarked
        )
        prediction = int(selected_model.predict(passenger)[0])
        if prediction == 1:
            st.success(f"{selected_model_name} predicts this passenger survived.")
        else:
            st.warning(f"{selected_model_name} predicts this passenger did not survive.")

        if hasattr(selected_model, "predict_proba"):
            probabilities = selected_model.predict_proba(passenger)[0]
            survival_probability = probabilities[list(selected_model.classes_).index(1)]
            st.metric("Model-estimated survival probability", f"{survival_probability:.1%}")
        else:
            st.info("Probability estimates are not enabled for this saved SVM model.")

with data_tab:
    st.subheader("Titanic passenger records")
    st.dataframe(passengers, width="stretch", hide_index=True)
