from flask import Flask, request, jsonify
import flask
import tensorflow as tf
import pandas as pd
import numpy as np
import json

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


loaded_model = tf.keras.models.load_model('saved_model/chatbot-brains-model')

app = Flask(__name__)



# import our chat-bot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']


for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique stemmed words", words)


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))



def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = loaded_model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    
    return return_list


@app.route("/predict", methods=['POST'])
def predictor():

    try:
        json_data = flask.request.json
        sentence = json_data['sentence']

        prediction = str(classify_local(sentence)).strip('[]')
        prediction_tuple = eval(prediction)
        print(prediction_tuple)
        print(type(prediction_tuple))
        
        word = prediction_tuple[0]
        probability = prediction_tuple[1]

        response_obj = {'status':'success',
                            "word":word,
                            "probability":probability
                            }
        return jsonify(response_obj)

            
    except Exception as error:
        logging.error("Error trying BACnet Read {}".format(error))
        info = str(error)
        print(error)
        response_obj = {'status':'fail','error':error}
        return jsonify(response_obj), 500

 


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
