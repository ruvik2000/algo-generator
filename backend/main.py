from flask import Flask, request
from flask_restful import Api, Resource
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import sklearn.preprocessing


app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def get(self, query):
        return {"data": query}

    def post(self):

        # Load the tokenizer for questions
        with open("tokeniser/question_tokenizer.pkl", "rb") as f:
            question_tokenizer = pickle.load(f)

        # Load the tokenizer for sequence representation
        with open("tokeniser/sequence_representation_tokenizer.pkl", "rb") as f:
            sequence_representation_tokenizer = pickle.load(f)

        # Load the algorithm label encoder
        with open("tokeniser/algorithm_encoder.pkl", "rb") as f:
            algorithm_encoder = pickle.load(f)

        print(request.form['query'])
        algorithm_model = load_model("algo_model.h5")
        new_question = request.form['query']
        new_question_sequence = question_tokenizer.texts_to_sequences([new_question])
        new_question_padded = pad_sequences(new_question_sequence, maxlen=29)

        predicted_algorithm_one_hot = algorithm_model.predict(new_question_padded)
        predicted_algorithm_index = np.argmax(predicted_algorithm_one_hot)
        predicted_algorithm = algorithm_encoder.inverse_transform([predicted_algorithm_index])

        print("Predicted Algorithm:")
        print(predicted_algorithm[0])

        return {"data": "hello world"}

api.add_resource(Prediction, "/prediction")

if __name__ == "__main__":
    app.run(debug=True)