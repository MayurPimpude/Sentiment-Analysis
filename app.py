from flask import Flask,render_template,request
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.models import load_model

app = Flask(__name__)

sent_length=20
voc_size=48040
model = load_model('data/my_model.h5')
model.summary()

@app.route('/')
def web():
    return render_template('home.html',message=0,messag=0)


@app.route('/predict', methods=['POST','GET'])
def home():
    
        if request.method == "POST":
            message = request.form.get("message")
            
            s=[one_hot(words,voc_size) for words in message]
            print(s)

            padded = pad_sequences(s,padding='pre',maxlen=sent_length)
            print(padded)

            sentiment = model.predict([[padded]])
            print(sentiment[0])

            sentiment = sentiment[0]
            sentiment1 = float(sentiment)
            print(sentiment1)

            return render_template('home.html',message=sentiment1,messag=1-sentiment1)
 

if __name__ == '__main__':
    app.run(debug=True)