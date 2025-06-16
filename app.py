from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def grafik():
    return render_template('grafik.html')  # akan ambil dari folder templates

if __name__ == '__main__':
    app.run(debug=True)


# === Load dan Training Model Sekali Saat Start ===

""" # Load dataset
df = pd.read_csv('marketing_campaign_cleaned.csv')

# Label Encoding untuk 'Education' dan 'Marital_Status'
le_education = LabelEncoder()
le_marital = LabelEncoder()
df['Education'] = le_education.fit_transform(df['Education'])
df['Marital_Status'] = le_marital.fit_transform(df['Marital_Status'])

# Features dan Target
X = df.drop('Response', axis=1)
y = df['Response']

# Standardisasi Fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_scaled, y)

# List fitur yang diperlukan (nama kolom)
feature_names = list(X.columns)

@app.route('/')
def index():
    return "✅ Flask Decision Tree API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Pastikan semua fitur tersedia
    missing_features = [feature for feature in feature_names if feature not in data]
    if missing_features:
        return jsonify({'error': f'Missing features: {missing_features}'}), 400

    # Convert input ke DataFrame
    input_df = pd.DataFrame([data])

    # Encoding manual untuk Education dan Marital_Status
    input_df['Education'] = le_education.transform(input_df['Education'])
    input_df['Marital_Status'] = le_marital.transform(input_df['Marital_Status'])

    # Normalisasi data input
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0].tolist()

    return jsonify({
        'prediction': int(prediction),
        'probability': probability
    })

if __name__ == '__main__':
    app.run(debug=True)
 """