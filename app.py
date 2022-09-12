from flask import Flask,render_template,request
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import tensorflow as tf
#from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.models import load_model

app = Flask(__name__)

@app.route('/')
def web():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    sent_length=20
    voc_size=48040
    #model = pickle.load(open('data/my_model.h5', 'rb'))
    model = load_model('data/my_model.h5','rb')
    try:
        if request.method == "POST":
            message = request.form.get("message")

            s=[one_hot(words,voc_size) for words in message]
            padded = pad_sequences(s,padding='pre',maxlen=sent_length)

            sentiment = model.predict(padded)
            print('Positive Rate :',sentiment)

        return print("you are successfuly logged in"+message)
    except:
        print("error")
    return render_template('home.html')
 

if __name__ == '__main__':
    app.run(debug=True)