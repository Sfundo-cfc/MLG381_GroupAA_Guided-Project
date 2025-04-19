import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# Load the trained Keras model
model = load_model("student_performance_model.keras")

# Load the dataset for extracting structure
df = pd.read_csv("Student_performance_data.csv")
X = df.drop(['StudentID', 'GradeClass'], axis=1)
Y = df['GradeClass']

# One-hot encode categorical columns (fit same way as training)
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Encode labels
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Feature columns after encoding
feature_names = X_encoded.columns

# Prediction function
def predict_class(user_input_dict):
    input_df = pd.DataFrame([user_input_dict])
    input_encoded = pd.get_dummies(input_df)
    
    # Align columns with training data
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    
    prediction = model.predict(input_scaled)
    predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
    return predicted_class

# Create Dash app
app = dash.Dash(__name__)
server = app.server  # for Render deployment

app.layout = html.Div([
    html.H1("Student Grade Class Predictor", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Age"),
        dcc.Input(id='Age', type='number', value=18),

        html.Label("Gender"),
        dcc.Dropdown(id='Gender', options=[
            {'label': g, 'value': g} for g in df['Gender'].unique()
        ], value=df['Gender'].unique()[0]),

        html.Label("Ethnicity"),
        dcc.Dropdown(id='Ethnicity', options=[
            {'label': e, 'value': e} for e in df['Ethnicity'].unique()
        ], value=df['Ethnicity'].unique()[0]),

        html.Label("Parental Education"),
        dcc.Dropdown(id='ParentalEducation', options=[
            {'label': p, 'value': p} for p in df['ParentalEducation'].unique()
        ], value=df['ParentalEducation'].unique()[0]),

        html.Label("Study Time Weekly"),
        dcc.Input(id='StudyTimeWeekly', type='number', value=5),

        html.Label("Absences"),
        dcc.Input(id='Absences', type='number', value=0),

        html.Label("Tutoring"),
        dcc.Dropdown(id='Tutoring', options=[{'label': x, 'value': x} for x in df['Tutoring'].unique()], value=df['Tutoring'].unique()[0]),

        html.Label("Parental Support"),
        dcc.Dropdown(id='ParentalSupport', options=[{'label': x, 'value': x} for x in df['ParentalSupport'].unique()], value=df['ParentalSupport'].unique()[0]),

        html.Label("Extracurricular"),
        dcc.Dropdown(id='Extracurricular', options=[{'label': x, 'value': x} for x in df['Extracurricular'].unique()], value=df['Extracurricular'].unique()[0]),

        html.Label("Sports"),
        dcc.Dropdown(id='Sports', options=[{'label': x, 'value': x} for x in df['Sports'].unique()], value=df['Sports'].unique()[0]),

        html.Label("Music"),
        dcc.Dropdown(id='Music', options=[{'label': x, 'value': x} for x in df['Music'].unique()], value=df['Music'].unique()[0]),

        html.Label("Volunteering"),
        dcc.Dropdown(id='Volunteering', options=[{'label': x, 'value': x} for x in df['Volunteering'].unique()], value=df['Volunteering'].unique()[0]),

        html.Label("GPA"),
        dcc.Input(id='GPA', type='number', value=3.0, step=0.1),

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
    print("app.py is running!")

    app.run(debug=True, host="0.0.0.0", port=8080)#any device on same network can access the app using the server's IP address and port 8080
    # app.run_server(debug=True, port=8050)  # for local testing

