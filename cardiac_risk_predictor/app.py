from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and prepare the data (This will be used to train the model)
df = pd.read_csv("framingham.csv")
df.drop(['education'], axis=1, inplace=True)
df.rename(columns={'male': 'Sex_male'}, inplace=True)
df.dropna(inplace=True)

features = ['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']
X = df[features]
y = df['TenYearCHD']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get user input from the form
            age = float(request.form['age'])
            sex_male = int(request.form['sex_male'])
            cigs_per_day = float(request.form['cigs_per_day'])
            tot_chol = float(request.form['tot_chol'])
            sys_bp = float(request.form['sys_bp'])
            glucose = float(request.form['glucose'])

            # Prepare the input for prediction
            user_input = np.array([[age, sex_male, cigs_per_day, tot_chol, sys_bp, glucose]])
            user_input_scaled = scaler.transform(user_input)

            # Get prediction and probability
            prediction = model.predict(user_input_scaled)[0]
            probability = model.predict_proba(user_input_scaled)[0][1] * 100

            result = {
                'prediction': prediction,
                'probability': probability
            }

            return render_template('index.html', result=result)
        except ValueError:
            return render_template('index.html', error="❌ Invalid input. Please enter numeric values.")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
