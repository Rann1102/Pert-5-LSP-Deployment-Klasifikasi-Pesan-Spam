import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import streamlit as st

# Memuat dataset
data_path = 'sms_spam_indo.csv' 
df = pd.read_csv(data_path)

# Clean the message text
def clean_text(text):
   text = re.sub(r'\W', ' ', text)  # Menghapus karakter non-alphanumeric
   text = re.sub(r'\s+', ' ', text)  # Menghapus spasi ekstra
   text = text.lower().strip()       # Mengubah menjadi huruf kecil
   return text

# Menerapkan pembersihan pada kolom pesan
df['Pesan_cleaned'] = df['Pesan'].apply(clean_text)

# Memisahkan data menjadi set pelatihan dan pengujian
X = df['Pesan_cleaned']
y = df['Kategori']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengonversi data teks menjadi vektor numerik menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Menginisialisasi dan melatih model regresi logistik
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Menyimpan model terlatih dan vectorizer TF-IDF
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'Classification Spam or Ham.pkl')

st.title('SMS Spam Classifier')

# Input teks
message = st.text_area('Masukkan pesan SMS')

# Memuat model 
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('Classification Spam or Ham.pkl')

# Membersihkan dan mengonversi pesan input
if message:
    cleaned_message = clean_text(message)
    message_tfidf = vectorizer.transform([cleaned_message])

    # Melakukan Prediksi
    prediction = model.predict(message_tfidf)

    # Menampilkan hasil
    if prediction == 'spam':
        st.error('Pesan ini diklasifikasikan sebagai SPAM.')
    else:
        st.success('Pesan ini diklasifikasikan sebagai HAM.')

# Untuk evaluasi (opsional)
if st.button('Evaluate Model'):
    y_pred = model.predict(vectorizer.transform(X_test))
    st.text(classification_report(y_test, y_pred))
