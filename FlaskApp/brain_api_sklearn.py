from flask import Flask, request, jsonify
import flask

from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np
import json
from operator import itemgetter

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


app = Flask(__name__)


'''
START BY PREPPING TRAINING DATA intents.json file
'''

# TRAINING DATA
with open('hvac_babble.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']


for (category, q_a) in hvac_world.items():
    # tokenize each word in the sentence
    w = nltk.word_tokenize(q_a['answer'][0])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, category))
    # add to our classes list
    if category not in classes:
        classes.append(category)


# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), 'documents')
# classes = intents
print(len(classes), 'classes', classes)
# words = all words, vocabulary
print(len(words), 'unique stemmed words', words)


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
#random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])


# Fit the sklearn model
model = MLPClassifier(learning_rate_init=0.0001,max_iter=9000,shuffle=True).fit(train_x, train_y)


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
                    print ('found in bag: %s' % w)
    return(np.array(bag))

def classify_local(sentence):
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])

    results = model.predict_proba(input_data)[0]

    results =  np.round(results,4).tolist()
    probs_and_classes = list(zip(classes, results))

    print(probs_and_classes)

    best_result = max(probs_and_classes,key=itemgetter(1))

    # return tuple of intent and probability
    return best_result


@app.route('/predict', methods=['POST'])
def predictor():

    try:
        json_data = flask.request.json
        sentence = json_data['sentence']

        prediction = str(classify_local(sentence)).strip('[]')

        prediction_tuple = eval(prediction)
        tag = prediction_tuple[0]
        probability = prediction_tuple[1]

        get_answer = hvac_world.get(tag, None)
        bot_reply = get_answer['answer']

        response_obj = {'status':'success',
                            'tag':tag,
                            'probability':probability,
                            'bot_reply':bot_reply}
        return jsonify(response_obj)


    except Exception as e:
        info = f'Classification Error: {e}'
        print(info)
        response_obj = {'status':'fail','error':info}
        return jsonify(response_obj), 500


@app.route('/')
def hello_world():
    message = 'Hello from Flask!'
    response_obj = {'status':'success','message':message}
    return jsonify(response_obj)



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
