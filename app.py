# WORKING CODE (BACKUP)----------------------------------------------------------------------------------
# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# import pickle

# app = Flask(__name__)
# model = pickle.load(open('heart-disease-prediction-logreg-model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler used during training
# feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))  # Load the feature columns used during training

# @app.route('/')
# def home():
#     return render_template('main.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get form data
#         age = int(request.form['age'])
#         sex = int(request.form.get('sex'))  # Assuming 1 for male, 0 for female
#         cp = int(request.form.get('cp'))
#         trestbps = int(request.form['trestbps'])
#         chol = int(request.form['chol'])
#         fbs = int(request.form.get('fbs'))  # Assuming 1 for true, 0 for false
#         restecg = int(request.form['restecg'])
#         thalach = int(request.form['thalach'])
#         exang = int(request.form.get('exang'))  # Assuming 1 for yes, 0 for no
#         oldpeak = float(request.form['oldpeak'])
#         slope = int(request.form.get('slope'))
#         ca = int(request.form['ca'])
#         thal = int(request.form.get('thal'))  # Assuming numerical values for thal        

#         # Create a DataFrame for consistent preprocessing
#         input_data = pd.DataFrame([{
#             'age': age,
#             'sex': sex,
#             'chest pain type': cp,
#             'resting blood pressure': trestbps,
#             'serum cholestoral': chol,
#             'fasting blood sugar': fbs,
#             'resting electrocardiographic results': restecg,
#             'max heart rate': thalach,
#             'exercise induced angina': exang,
#             'oldpeak': oldpeak,
#             'ST segment': slope,
#             'major vessels': ca,
#             'thal': thal
#         }])

#         # Apply one-hot encoding to match the training data
#         cate_val = ['sex', 'chest pain type', 'fasting blood sugar', 'resting electrocardiographic results', 'exercise induced angina', 'ST segment', 'thal']
#         input_data = pd.get_dummies(input_data, columns=cate_val, drop_first=True)

#         # Manually add missing columns with zeros
#         for col in feature_columns:
#             if col not in input_data.columns:
#                 input_data[col] = 0

#         # Reorder columns to match the training data
#         input_data = input_data[feature_columns]

#         # Scale the continuous features
#         cont_val = ['age', 'resting blood pressure', 'serum cholestoral', 'max heart rate', 'oldpeak', 'major vessels']
#         input_data[cont_val] = scaler.transform(input_data[cont_val])

#         # Making predictions
#         prediction = model.predict(input_data)

#         # Generate health messages based on input values
#         my_prediction = []
        
#         if trestbps >= 94:
#             my_prediction.append("Your resting blood pressure is HIGHER than the usual!")
#         else:
#             my_prediction.append("Your resting blood pressure is within the biological reference interval.")
        
#         if chol >= 126:
#             my_prediction.append("Your serum cholestoral is HIGHER than the usual!")
#         else:
#             my_prediction.append("Your serum cholestoral is within the biological reference interval.")
        
#         if thalach >= 96:
#             my_prediction.append("Your maximum heart rate is HIGHER than the usual!")
#         else:
#             my_prediction.append("Your maximum heart rate is within the biological reference interval.")
        
#         return render_template('result.html', prediction=prediction[0], messages=my_prediction)

# if __name__ == '__main__':
#     app.run(debug=True)
#------------------------------------------------------------------------------------------------------------
# from flask import Flask, render_template, request
# import pandas as pd
# import pickle

# app = Flask(__name__)
# model = pickle.load(open('heart-disease-prediction-tree-model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler used during training
# feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))  # Load the feature columns used during training

# def get_dietary_recommendations(metrics):
#     min_values = {
#         'resting blood pressure': 120,
#         'serum cholestoral': 126,
#         'max heart rate': 100,
#         'resting electrocardiographic results': 0,
#         'oldpeak': 0.0,
#         'major vessels': 0,
#     }
    
#     dietary_recommendations = {
#         'resting blood pressure': 'Reduce salt intake, eat more fruits and vegetables.',
#         'serum cholestoral': 'Avoid fatty foods, increase fiber intake.',
#         'max heart rate': 'Engage in regular cardiovascular exercise, manage stress.',
#         'resting electrocardiographic results': 'Consult a healthcare provider for specific dietary recommendations, avoid stimulants like caffeine.',
#         'oldpeak': 'Follow a heart-healthy diet, manage stress levels.',
#         'major vessels': 'Follow a balanced diet, maintain regular physical activity.',
#     }

#     recommendations = []
#     for metric, value in metrics.items():
#         if value >= min_values[metric]:
#             recommendations.append(f"{metric}: {dietary_recommendations[metric]}")

#     return recommendations

# @app.route('/')
# def home():
#     return render_template('main.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get form data
#         age = int(request.form['age'])
#         sex = int(request.form.get('sex'))  # Assuming 1 for male, 0 for female
#         cp = int(request.form.get('cp'))
#         trestbps = int(request.form['trestbps'])
#         chol = int(request.form['chol'])
#         fbs = int(request.form.get('fbs'))  # Assuming 1 for true, 0 for false
#         restecg = int(request.form['restecg'])
#         thalach = int(request.form['thalach'])
#         exang = int(request.form.get('exang'))  # Assuming 1 for yes, 0 for no
#         oldpeak = float(request.form['oldpeak'])
#         slope = int(request.form.get('slope'))
#         ca = int(request.form['ca'])
#         thal = int(request.form.get('thal'))  # Assuming numerical values for thal        

#         # Create a DataFrame for consistent preprocessing
#         input_data = pd.DataFrame([{
#             'age': age,
#             'sex': sex,
#             'chest pain type': cp,
#             'resting blood pressure': trestbps,
#             'serum cholestoral': chol,
#             'fasting blood sugar': fbs,
#             'resting electrocardiographic results': restecg,
#             'max heart rate': thalach,
#             'exercise induced angina': exang,
#             'oldpeak': oldpeak,
#             'ST segment': slope,
#             'major vessels': ca,
#             'thal': thal
#         }])

#         # Apply one-hot encoding to match the training data
#         cate_val = ['sex', 'chest pain type', 'fasting blood sugar', 'resting electrocardiographic results', 'exercise induced angina', 'ST segment', 'thal']
#         input_data = pd.get_dummies(input_data, columns=cate_val, drop_first=True)

#         # Manually add missing columns with zeros
#         for col in feature_columns:
#             if col not in input_data.columns:
#                 input_data[col] = 0

#         # Reorder columns to match the training data
#         input_data = input_data[feature_columns]

#         # Scale the continuous features
#         cont_val = ['age', 'resting blood pressure', 'serum cholestoral', 'max heart rate', 'oldpeak', 'major vessels']
#         input_data[cont_val] = scaler.transform(input_data[cont_val])

#         # Making predictions
#         prediction = model.predict(input_data)

#         # Generate health messages and dietary recommendations based on input values
#         health_metrics = {
#             'resting blood pressure': trestbps,
#             'serum cholestoral': chol,
#             'max heart rate': thalach
#         }

#         health_messages = []

#         if trestbps >= 94:
#             health_messages.append("Your resting blood pressure is HIGHER than the usual!")
#         else:
#             health_messages.append("Your resting blood pressure is within the biological reference interval.")
        
#         if chol >= 126:
#             health_messages.append("Your serum cholestoral is HIGHER than the usual!")
#         else:
#             health_messages.append("Your serum cholestoral is within the biological reference interval.")
        
#         if thalach >= 96:
#             health_messages.append("Your maximum heart rate is HIGHER than the usual!")
#         else:
#             health_messages.append("Your maximum heart rate is within the biological reference interval.")
        
#         dietary_recommendations = get_dietary_recommendations(health_metrics)
        
#         return render_template('result.html', prediction=prediction[0], messages=health_messages, recommendations=dietary_recommendations)

# if __name__ == '__main__':
#     app.run(debug=True)


# ACTIVE CODE ----------------------------------------------------------------------------------------------
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('heart-disease-prediction-logreg-model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler used during training
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))  # Load the feature columns used during training

def get_dietary_recommendations(metrics):
    min_values = {
        'resting blood pressure': 120,
        'serum cholestoral': 126,
        'max heart rate': 100,
        #'fasting blood sugar': 100,
        # 'age': 29,
        # 'sex': 0,
        #'chest pain type': 1,
        'resting electrocardiographic results': 0,
        #'exercise induced angina': 0,
        'oldpeak': 0.0,
        #'ST segment': 1,
        'major vessels': 0,
        #'thal': 3
    }
    
    dietary_recommendations = {
        'resting blood pressure': 'Reduce salt intake, eat more fruits and vegetables.',
        'serum cholestoral': 'Avoid fatty foods, increase fiber intake.',
        'max heart rate': 'Engage in regular cardiovascular exercise, manage stress.',
        #'fasting blood sugar': 'Reduce sugar intake, eat more whole grains and vegetables.',
        # 'age': 'Maintain a balanced diet with plenty of fruits, vegetables, and proteins.',
        # 'sex': 'Ensure adequate protein intake, focus on heart-healthy foods.',
        #'chest pain type': 'Follow a heart-healthy diet, avoid heavy meals.',
        'resting electrocardiographic results': 'Consult a healthcare provider for specific dietary recommendations, avoid stimulants like caffeine.',
        #'exercise induced angina': 'Adopt a heart-healthy diet, avoid heavy meals before exercise.',
        'oldpeak': 'Follow a heart-healthy diet, manage stress levels.',
        #'ST segment': 'Reduce salt and saturated fat intake, eat more fruits and vegetables.',
        'major vessels': 'Follow a balanced diet, maintain regular physical activity.',
        #'thal': 'Maintain a balanced diet, focus on heart-healthy foods.'
    }

    recommendations = []
    for metric, value in metrics.items():
        if value >= min_values[metric]:
            recommendations.append(f"{metric}: {dietary_recommendations[metric]}")

    return recommendations

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = int(request.form.get('sex'))  # Assuming 1 for male, 0 for female
        cp = int(request.form.get('cp'))
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form.get('fbs'))  # Assuming 1 for true, 0 for false
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form.get('exang'))  # Assuming 1 for yes, 0 for no
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form.get('slope'))
        ca = int(request.form['ca'])
        thal = int(request.form.get('thal'))  # Assuming numerical values for thal        

        # Create a DataFrame for consistent preprocessing
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'chest pain type': cp,
            'resting blood pressure': trestbps,
            'serum cholestoral': chol,
            'fasting blood sugar': fbs,
            'resting electrocardiographic results': restecg,
            'max heart rate': thalach,
            'exercise induced angina': exang,
            'oldpeak': oldpeak,
            'ST segment': slope,
            'major vessels': ca,
            'thal': thal
        }])

        # Apply one-hot encoding to match the training data
        cate_val = ['sex', 'chest pain type', 'fasting blood sugar', 'resting electrocardiographic results', 'exercise induced angina', 'ST segment', 'thal']
        input_data = pd.get_dummies(input_data, columns=cate_val, drop_first=True)

        # Manually add missing columns with zeros
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match the training data
        input_data = input_data[feature_columns]

        # Scale the continuous features
        cont_val = ['age', 'resting blood pressure', 'serum cholestoral', 'max heart rate', 'oldpeak', 'major vessels']
        input_data[cont_val] = scaler.transform(input_data[cont_val])

        # Making predictions
        prediction = model.predict(input_data)

        # Generate health messages and dietary recommendations based on input values
        health_metrics = {
            'resting blood pressure': trestbps,
            'serum cholestoral': chol,
            'max heart rate': thalach
        }

        health_messages = []

        if trestbps >= 94:
            health_messages.append("Your resting blood pressure is HIGHER than the usual!")
        else:
            health_messages.append("Your resting blood pressure is within the biological reference interval.")
        
        if chol >= 126:
            health_messages.append("Your serum cholestoral is HIGHER than the usual!")
        else:
            health_messages.append("Your serum cholestoral is within the biological reference interval.")
        
        if thalach >= 96:
            health_messages.append("Your maximum heart rate is HIGHER than the usual!")
        else:
            health_messages.append("Your maximum heart rate is within the biological reference interval.")
        
        dietary_recommendations = get_dietary_recommendations(health_metrics)
        
        return render_template('result.html', prediction=prediction[0], messages=health_messages, recommendations=dietary_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
