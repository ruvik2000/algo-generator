# from flask import Flask, request
# from flask_restful import Api, Resource
# from keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import pickle
# import sklearn.preprocessing


# app = Flask(__name__)
# api = Api(app)

# class Prediction(Resource):
#     def get(self, query):
#         return {"data": query}

#     def post(self):

#         # Load the tokenizer for questions
#         with open("tokeniser/question_tokenizer.pkl", "rb") as f:
#             question_tokenizer = pickle.load(f)

#         # Load the tokenizer for sequence representation
#         with open("tokeniser/sequence_representation_tokenizer.pkl", "rb") as f:
#             sequence_representation_tokenizer = pickle.load(f)

#         # Load the algorithm label encoder
#         with open("tokeniser/algorithm_encoder.pkl", "rb") as f:
#             algorithm_encoder = pickle.load(f)

#         print(request.form['query'])
#         algorithm_model = load_model("algo_model.h5")
#         new_question = request.form['query']
#         new_question_sequence = question_tokenizer.texts_to_sequences([new_question])
#         new_question_padded = pad_sequences(new_question_sequence, maxlen=29)

#         predicted_algorithm_one_hot = algorithm_model.predict(new_question_padded)
#         predicted_algorithm_index = np.argmax(predicted_algorithm_one_hot)
#         predicted_algorithm = algorithm_encoder.inverse_transform([predicted_algorithm_index])

#         print("Predicted Algorithm:")
#         print(predicted_algorithm[0])

#         return {"data": "hello world"}

# api.add_resource(Prediction, "/prediction")

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify  #, jsonify
import tensorflow as tf
from flask_restful import Api, Resource
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from graphviz import Digraph
import re

app = Flask(__name__)
api = Api(app)
CORS(app)

# Load models and tokenizers
model_algorithm = load_model("algo_model.h5")
model_sequence = load_model("seq_model.h5")

# Load the tokenizer for questions
with open("tokeniser/question_tokenizer.pkl", "rb") as f:
    question_tokenizer = pickle.load(f)

# Load the tokenizer for sequence representation
with open("tokeniser/sequence_representation_tokenizer.pkl", "rb") as f:
    sequence_representation_tokenizer = pickle.load(f)

# Load the algorithm label encoder
with open("tokeniser/algorithm_encoder.pkl", "rb") as f:
    algorithm_encoder = pickle.load(f)

def get_predictions(question):
    # Preprocess the input question
    question_sequence = question_tokenizer.texts_to_sequences([question])
    question_padded = pad_sequences(question_sequence, maxlen=27)

    # Predict the algorithm
    predicted_algorithm_one_hot = model_algorithm.predict(question_padded)
    predicted_algorithm_index = np.argmax(predicted_algorithm_one_hot)
    predicted_algorithm = algorithm_encoder.inverse_transform([predicted_algorithm_index])[0]

    # Predict the sequence representation
    sequence_representation_prediction = model_sequence.predict(question_padded)
    sequence_representation_indices = np.argmax(sequence_representation_prediction, axis=-1)
    predicted_sequence_representation = ' '.join(sequence_representation_tokenizer.index_word[idx] for idx in sequence_representation_indices[0] if idx > 0)

    return predicted_algorithm, predicted_sequence_representation

def get_diagram(sequence):
    # Convert sequence representation to a list of strings
    sequence_list = sequence.split()

    # Initialize the Graphviz graph
    flowchart = Digraph("Flowchart", format="png")
    flowchart.attr(rankdir="TB", size="10")

    # Initialize variables to track the current and previous nodes
    previous_node = None

    # Loop through the sequence list
    for index, item in enumerate(sequence_list):
        # Create nodes for processes and decisions
        if item.startswith("<process:") or item.startswith("<decision:"):
            node_label = re.sub(r'[<>]', '', item)
            flowchart.node(str(index), label=node_label)
            current_node = str(index)

            # Connect the current node to the previous node
            if previous_node is not None:
                flowchart.edge(previous_node, current_node)
            previous_node = current_node

        # Create nodes and edges for loop and end conditions
        elif item == "<loop>":
            flowchart.edge(current_node, previous_node, label="Loop")
            previous_node = None
        elif item == "<true>":
            flowchart.edge(current_node, str(index + 1), label="True")
        elif item == "<false>":
            flowchart.edge(current_node, str(index + 1), label="False")
        elif item == "<end>":
            if previous_node is not None:
                flowchart.edge(previous_node, current_node)
                previous_node = current_node
            else:
                previous_node = current_node
            flowchart.node("End", label="End")
            flowchart.edge(previous_node, "End")
            previous_node = "End"

    # Save the flowchart as a PNG image
    # flowchart.render("flowchart_output", cleanup=True)

    png_binary_data = flowchart.pipe(format='png')
    print(png_binary_data)
    

class Prediction(Resource):
    def get(self, query):
        return {"data": query}

    def post(self):
        data = request.get_json()
        query = data['query']
        predicted_algorithm, predicted_sequence_representation = get_predictions(query)
        print("predicted_algorithm : ", predicted_algorithm)
        print("predicted_sequence_representation : ", predicted_sequence_representation)
        # get_diagram(predicted_sequence_representation)
        response = {'algo_prediction': predicted_algorithm}
        return jsonify(response)

api.add_resource(Prediction, "/prediction")

if __name__ == "__main__":
    app.run(debug=True)
