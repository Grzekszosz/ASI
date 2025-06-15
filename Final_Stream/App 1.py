import streamlit as st
import pandas as pd
import requests
import os
import pickle  # lub pickle
from AzureBlob import download_model_from_azure


# ------------------ Ściąganie modelu z GitHub ------------------
MODEL_URL = "https://raw.githubusercontent.com/Grzekszosz/ASI_model/main/ag-20250517_103107/models/CatBoost/model.pkl"
MODEL_PATH = "model.pkl"

@st.cache_data
def download_model():
    if not os.path.exists("model.joblib"):
        download_model_from_azure("container", "model.pkl", "model.pkl")

    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


# ------------------ Funkcje pomocnicze ------------------
def generate_datetime_features(date: pd.Timestamp) -> dict:
    return {
        'year_num': date.year,
        'month_num': date.month,
        'dayofweek_num': date.dayofweek,
        'dayofmonth': date.day,
        'dayofyear_num': date.day_of_year
    }

def generate_text_features(subject: str, message: str) -> dict:
    full_text = subject + " " + message

    total_letters = sum(c.isalpha() for c in full_text)
    uppercase_letters = sum(c.isupper() for c in full_text)
    lowercase_letters = sum(c.islower() for c in full_text)
    special_digits = sum(not c.isalpha() for c in full_text)

    ratio_upper_case = round(uppercase_letters / lowercase_letters, 2) if lowercase_letters > 0 else 0
    ratio_upper_and_special_case = round(total_letters / special_digits, 2) if special_digits > 0 else 0

    return {
        'ratio_upper_case': ratio_upper_case,
        'ratio_upper_and_special_case': ratio_upper_and_special_case,
        'Subject': subject,
        'Message': message,
        'Date': str(pd.Timestamp.now())  # lub użyj date z formularza
    }

# ------------------ Streamlit GUI ------------------
st.set_page_config(page_title="Email Feature Extractor", page_icon="📧")

st.title("📧 Email Feature Extractor")
st.write("Wprowadź dane e-maila, a system wygeneruje cechy potrzebne do analizy i sprawdzi, czy to spam.")

# Formularz
subject = st.text_input("📝 Temat wiadomości (Subject)")
message = st.text_area("💬 Treść wiadomości (Message)", height=200)
date = st.date_input("📅 Data otrzymania wiadomości (Date)")

if st.button("🔍 Wygeneruj cechy i sprawdź spam"):
    if not subject.strip() or not message.strip():
        st.warning("⚠️ Uzupełnij temat i treść wiadomości.")
    else:
        # Generowanie cech
        datetime_features = generate_datetime_features(pd.to_datetime(date))
        text_features = generate_text_features(subject, message)

        # Przygotowanie danych
        full_data = {
            "subject": subject,
            "message": message,
            "date": str(date)
        }
        full_data.update(datetime_features)
        full_data.update(text_features)

        st.success("✅ Cechy zostały wygenerowane!")

        # Wyświetlanie danych
        st.markdown("## 📥 Dane wejściowe")
        st.code(subject, language='plaintext')
        st.code(message, language='plaintext')
        st.write(f"**📅 Data:** `{date}`")

        st.markdown("---")
        st.markdown("## ⚙️ Wygenerowane cechy")

        features_only = {k: v for k, v in full_data.items() if k not in ['subject', 'message', 'date']}
        for key, value in features_only.items():
            st.write(f"- **{key}**: {value}")

        # Pobieranie modelu
        try:
            model = download_model()
        except Exception as e:
            st.error(f"❌ Błąd podczas pobierania lub ładowania modelu: {e}")
            st.stop()

        # Klasyfikacja
        feature_df = pd.DataFrame([features_only])
        categorical_cols = model.cat_feature_names
        for col in categorical_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype('category')

        st.write("Twoje feature_df.columns:", feature_df.columns.tolist())
        prediction = model.predict(feature_df)[0]
        prediction_proba = model.predict_proba(feature_df)[0]

        st.markdown("## 🔍 Wynik klasyfikacji")
        if prediction == 1:
            st.error(f"📛 To może być SPAM (prawdopodobieństwo: {round(prediction_proba[1]*100, 2)}%)")
        else:
            st.success(f"📬 To wygląda na NIE-SPAM (prawdopodobieństwo: {round(prediction_proba[0]*100, 2)}%)")

        # Przycisk pobrania CSV
        csv = pd.DataFrame([full_data]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Pobierz wszystkie dane jako CSV",
            data=csv,
            file_name='email_features.csv',
            mime='text/csv'
        )
