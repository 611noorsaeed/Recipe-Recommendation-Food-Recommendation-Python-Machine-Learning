from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)



# Absolute path example
data = pd.read_csv("Notebook/recipe_final (1).csv")


# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

# Combine Features
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

def recommend_recipes(input_features):
    input_features_scaled = scaler.transform([input_features[:7]])
    input_ingredients_transformed = vectorizer.transform([input_features[7]])
    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
    distances, indices = knn.kneighbors(input_combined)
    recommendations = data.iloc[indices[0]]
    return recommendations[['recipe_name', 'ingredients_list', 'image_url']].head(5)

# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        carbohydrates = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        fiber = float(request.form['fiber'])
        ingredients = request.form['ingredients']
        input_features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
        recommendations = recommend_recipes(input_features)
        return render_template('index.html', recommendations=recommendations.to_dict(orient='records'),truncate = truncate)
    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)
