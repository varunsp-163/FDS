from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route("/")
def index():
    return render_template("webpage.html")


@app.route("/predict", methods=["POST"])    
def predict():
    if request.method == "POST":
        likes = float(request.form['likes'])
        saves = float(request.form['saves'])
        comments = float(request.form['comments'])
        shares = float(request.form['shares'])
        profile_visits = float(request.form['profile_visits'])
        follows = float(request.form['follows'])

        # Here you would use your model to make a prediction using the input features
        # For now, let's just return a dummy prediction
        features = np.array([[likes,saves,comments,shares, profile_visits,follows]])
        prediction = model.predict(features)

        return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
