from flask import Flask, request, jsonify
import flask

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import pandas as pd
import numpy as np
import json
import random

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


#modelo = tf.keras.models.load_model('saved_model/chatbot-brains-model')

app = Flask(__name__)




'''
START BY PREPPING TRAINING DATA intents.json file
'''
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
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique stemmed words", words)


'''
NEXT DEFINE DEEP LEARN MODEL AND TRAIN IT
'''
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])


# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
modelo = Sequential()
modelo.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(train_y[0]), activation='softmax'))


# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
modelo.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)


'''
HELPER FUNCTIONS BELOW
'''
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
    results = modelo.predict([input_data])[0]
    
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
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
        
        tag = prediction_tuple[0]
        probability = prediction_tuple[1]

        response_obj = {'status':'success',
                            "tag":tag,
                            "probability":probability
                            }
        return jsonify(response_obj)

            
    except Exception as e:
        info = f'Classification Error: {e}'
        print(info)
        response_obj = {'status':'fail','error':info}
        return jsonify(response_obj), 500

 


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)