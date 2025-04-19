import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# Load minimal data sample
df_sample = pd.read_csv("Student_performance_data.csv", nrows=50)
X_sample = df_sample.drop(['StudentID', 'GradeClass'], axis=1)
Y_sample = df_sample['GradeClass']

# Only keep categorical column names and dropdown values
categorical_cols = X_sample.select_dtypes(include=['object']).columns
dropdown_options = {col: [{'label': val, 'value': val} for val in df_sample[col].unique()] for col in categorical_cols}

# Precompute feature names and scalers using small sample
X_encoded = pd.get_dummies(X_sample, columns=categorical_cols, drop_first=True)
scaler = StandardScaler()
scaler.fit(X_encoded)
feature_names = X_encoded.columns

# Label encoder
le = LabelEncoder()
le.fit(Y_sample)

# Lazy-load model
model = None
def get_model():
    global model
    if model is None:
        model = load_model("student_performance_model.h5")
    return model

# Prediction logic
def predict_class(user_input_dict):
    model = get_model()
    input_df = pd.DataFrame([user_input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)
    predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
    return predicted_class

# Create app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Student Grade Class Predictor", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Age"), dcc.Input(id='Age', type='number', value=18),
        html.Label("Gender"), dcc.Dropdown(id='Gender', options=dropdown_options['Gender']),
        html.Label("Ethnicity"), dcc.Dropdown(id='Ethnicity', options=dropdown_options['Ethnicity']),
        html.Label("Parental Education"), dcc.Dropdown(id='ParentalEducation', options=dropdown_options['ParentalEducation']),
        html.Label("Study Time Weekly"), dcc.Input(id='StudyTimeWeekly', type='number', value=5),
        html.Label("Absences"), dcc.Input(id='Absences', type='number', value=0),
        html.Label("Tutoring"), dcc.Dropdown(id='Tutoring', options=dropdown_options['Tutoring']),
        html.Label("Parental Support"), dcc.Dropdown(id='ParentalSupport', options=dropdown_options['ParentalSupport']),
        html.Label("Extracurricular"), dcc.Dropdown(id='Extracurricular', options=dropdown_options['Extracurricular']),
        html.Label("Sports"), dcc.Dropdown(id='Sports', options=dropdown_options['Sports']),
        html.Label("Music"), dcc.Dropdown(id='Music', options=dropdown_options['Music']),
        html.Label("Volunteering"), dcc.Dropdown(id='Volunteering', options=dropdown_options['Volunteering']),
        html.Label("GPA"), dcc.Input(id='GPA', type='number', value=3.0, step=0.1),

        html.Br(), html.Button('Predict', id='predict-btn', n_clicks=0),
        html.H3(id='prediction-output', style={'marginTop': 20})
    ], style={'width': '60%', 'margin': 'auto'})
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('Age', 'value'),
    State('Gender', 'value'),
    State('Ethnicity', 'value'),
    State('ParentalEducation', 'value'),
    State('StudyTimeWeekly', 'value'),
    State('Absences', 'value'),
    State('Tutoring', 'value'),
    State('ParentalSupport', 'value'),
    State('Extracurricular', 'value'),
    State('Sports', 'value'),
    State('Music', 'value'),
    State('Volunteering', 'value'),
    State('GPA', 'value')
)
def update_prediction(n_clicks, Age, Gender, Ethnicity, ParentalEducation,
                      StudyTimeWeekly, Absences, Tutoring, ParentalSupport,
                      Extracurricular, Sports, Music, Volunteering, GPA):
    if n_clicks > 0:
        input_dict = {
            "Age": Age,
            "Gender": Gender,
            "Ethnicity": Ethnicity,
            "ParentalEducation": ParentalEducation,
            "StudyTimeWeekly": StudyTimeWeekly,
            "Absences": Absences,
            "Tutoring": Tutoring,
            "ParentalSupport": ParentalSupport,
            "Extracurricular": Extracurricular,
            "Sports": Sports,
            "Music": Music,
            "Volunteering": Volunteering,
            "GPA": GPA
        }
        prediction = predict_class(input_dict)
        return f"Predicted Grade Class: {prediction}"
    return ""

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
