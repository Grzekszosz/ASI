import streamlit as st
import pandas as pd
import os
import py7zr
from azure.storage.blob import BlobServiceClient
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv
load_dotenv()
# ------------------ Konfiguracja ------------------
BLOB_NAME = "ag.7z"
LOCAL_ZIP = "ag.7z"
EXTRACT_DIR = "autogluon_model"
MODEL_SUBDIR = "ag-20250619_000809"
MODEL_PATH = os.path.join(EXTRACT_DIR, MODEL_SUBDIR)

# ------------------ Pobieranie i ładowanie modelu ------------------
@st.cache_resource
def download_and_extract_model():

    if not os.path.exists(MODEL_PATH):
        conn_str = os.getenv("CONNSTR")
        if not conn_str:
            raise RuntimeError("Brak zmiennej środowiskowej CONNSTR")

        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service_client.get_container_client("container")
        blob_client = container_client.get_blob_client(BLOB_NAME)

        with open(LOCAL_ZIP, "wb") as f:
            f.write(blob_client.download_blob().readall())

        with py7zr.SevenZipFile("ag.7z", mode='r') as archive:
            archive.extractall(path="autogluon_model")

    return TabularPredictor.load(MODEL_PATH)

# ------------------ Funkcje pomocnicze ------------------
def generate_datetime_features(date: pd.Timestamp) -> dict:
    return {
        'year_num': date.year,
        'month_num': date.month,
        'dayofweek_num': date.dayofweek,
        'dayofmonth': date.day,
        'dayofyear_num': date.day_of_year,

    }

def ratio_special_case(subject: str, message: str) -> float:
    full_text = subject + " " + message
    total_letters = sum(c.isalpha() for c in full_text)
    special_digits = sum(not c.isalpha() for c in full_text)
    return round(total_letters / special_digits, 2) if special_digits > 0 else 0

def generate_text_features(subject: str, message: str) -> dict:
    full_text = subject + " " + message

    total_letters = sum(c.isalpha() for c in full_text)
    uppercase_letters = sum(c.isupper() for c in full_text)
    lowercase_letters = sum(c.islower() for c in full_text)
    special_digits = sum(not c.isalpha() for c in full_text)

    ratio_special_case = round(total_letters / special_digits,2) if special_digits > 0 else 0
    ratio_upper_case = round(uppercase_letters / lowercase_letters, 2) if lowercase_letters > 0 else 0
    ratio_upper_and_special_case = round(total_letters / special_digits, 2) if special_digits > 0 else 0

    return {
        'ratio_upper_and_special_case': ratio_upper_and_special_case,
        'ratio_special_case': ratio_special_case,
        'ratio_upper_case': ratio_upper_case,

    }
def generate_sub_mess(subject: str, message: str)->dict:
    return  {
        'Subject': subject,
        'Message': message,
        # 'Date': str(pd.Timestamp.now())
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
        # datetime_features = generate_datetime_features(pd.to_datetime(date))
        text_features = generate_text_features(subject, message)
        submess=generate_sub_mess(subject, message)
        # Przygotowanie danych
        full_data = {
            "subject": subject,
            "message": message,
            "date": str(date)
        }

        full_data.update(text_features)
        # full_data.update(datetime_features)
        full_data.update(submess)
        st.success("✅ Cechy zostały wygenerowane!")

        # Wyświetlanie danych
        st.markdown("## 📥 Dane wejściowe")
        st.code(subject)
        st.code(message)
        st.write(f"**📅 Data:** `{date}`")

        st.markdown("---")
        st.markdown("## ⚙️ Wygenerowane cechy")
        features_only = {k: v for k, v in full_data.items() if k not in ['subject', 'message', 'date']}
        for key, value in features_only.items():
            st.write(f"- **{key}**: {value}")

        # Pobieranie modelu
        try:
            model = download_and_extract_model()
        except Exception as e:
            st.error(f"❌ Błąd podczas pobierania lub ładowania modelu: {e}")
            st.stop()

        # Klasyfikacja
        feature_df = pd.DataFrame([features_only])
        predictor = TabularPredictor.load(MODEL_PATH)
        # Kategoryzacja wg metadanych modelu
        cat_columns = predictor.feature_metadata.type_map_raw.get("categorical", [])
        for col in cat_columns:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype("category")

        # prediction = model.predict(feature_df, model='CatBoost')[0]
        prediction = model.predict(feature_df)[0]
        prediction_proba = model.predict_proba(feature_df)[0]
        full_proba = model.predict_proba(feature_df)
        st.write("Full proba output:", full_proba)
        st.write("Model expects features:")
        st.write(model.feature_metadata.get_features())
        st.write("We provide:")
        st.write(feature_df.columns)

        st.write("features_only keys:", features_only.keys())
        st.write("feature_df:", feature_df.head())
        st.write("model expects:", model.feature_metadata.get_features())
        st.write("📊 Raw prediction:", prediction)
        st.write("📊 Probabilities:", full_proba)
        st.write("📊 Model classes:", model.class_labels)
        st.write("📊 Positive class:", model.positive_class)
        st.write("📊 Leaderboard:")
        st.dataframe(model.leaderboard(silent=True))

        if isinstance(full_proba, pd.DataFrame):
            spam_proba = full_proba.iloc[0][1]
            ham_proba = full_proba.iloc[0][0]
        elif isinstance(full_proba, np.ndarray):
            spam_proba = full_proba[0][1]
            ham_proba = full_proba[0][0]
        elif isinstance(full_proba, float):
            # fallback – tylko dla modeli zwracających 1 liczbową wartość (np. prawdopodobieństwo klasy 1)
            spam_proba = full_proba
            ham_proba = 1 - spam_proba
        else:
            raise TypeError("Nieznany typ danych dla predict_proba")
        if prediction == 1:
            st.error(f"📛 To może być SPAM (prawdopodobieństwo: {round(spam_proba*100, 2)}%)")
        else:
            st.success(f"📬 To wygląda na NIE-SPAM (prawdopodobieństwo: {round(spam_proba*100, 2)}%)")

        # Przycisk pobrania CSV
        csv = pd.DataFrame([full_data]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Pobierz wszystkie dane jako CSV",
            data=csv,
            file_name='email_features.csv',
            mime='text/csv'
        )
