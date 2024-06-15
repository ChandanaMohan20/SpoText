import keras_core as keras
import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
from tensorflow import keras
from flask import Flask, request, render_template
import re
from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
import pandas as pd
#import tensorflow_text as tf_text
import unicodedata
#from imblearn.over_sampling import SMOTE


train_data = pd.read_csv('C:\\Users\\chandana mohan\\Desktop\\Pjt\\pjt\\.venv\\train_data.csv')

def Clean(text):
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.regex_replace(text, '\.\.\.', ' ')
    text = tf.strings.join(['',text, ''], separator=' ')
    return text

def clean_text(text):
    # Remove Twitter handles starting with '@'
    text = re.sub(r'@\w+', '', text)
    # Remove non-alphanumeric characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert multiple whitespace characters to a single space
    text = re.sub(r'\s+', ' ', text)
    # Convert the text to lowercase
    text = text.lower()
    return text

max_features = 75000
embedding_dim = 64
sequence_length = 1024



class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


app = Flask(__name__, template_folder='C:\\Users\\chandana mohan\\Desktop\\Pjt\\pjt\\.venv\\template')

def load_model_with_custom_objects(model_file="C:\\Users\\chandana mohan\\Desktop\\Pjt\\pjt\\.venv\\model.h5"):
    # Your function implementation here
   with tf.keras.utils.custom_object_scope({'TransformerBlock': TransformerBlock}):
    model = tf.keras.models.load_model(model_file, compile=False)
    return model
   
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')



@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        model_path = "C:\\Users\\chandana mohan\\Desktop\\Pjt\\pjt\\.venv\\model.h5"
      
        model = load_model_with_custom_objects(model_path)
        custom_text = request.form.get('InputData')
        custom_text1 = request.form.get('InputData')
        #tokenizer = Tokenizer()
        #sequences = tokenizer.texts_to_sequences([custom_text])
        #value = np.array([sequences])
        #padded_sequences = pad_sequences(value, maxlen=1024)
        #value = np.array([custom_text])
        #values = np.expand_dims(value, axis=1)
        max_features = 75000
        embedding_dim = 64
        sequence_length = 1024
        vectorize_layer = TextVectorization(
            standardize=Clean ,
            max_tokens=max_features,
            ngrams = (3,5),
            output_mode="int",
            output_sequence_length=sequence_length,
            pad_to_max_tokens=True
        )

        #train_data['text'] = train_data['text'].astype(str)
        vectorize_layer.adapt(train_data['text'].values)
        
        #vectorize_layer(train_data['text']).numpy()

        #preprocessed_text = vectorize_layer(tf.constant([clean_text(custom_text)]))

        custom_text = Clean(custom_text)
        custom_text_preprocessed = vectorize_layer([custom_text])
        #print(custom_text_preprocessed)

        
        
        

       

        

        # Tokenize and pad the input text
        #sequences = tokenizer.texts_to_sequences([custom_text_preprocessed])
        #padded_sequences = pad_sequences(value, maxlen=sequence)

        # Make predictions
        predictions = model.predict(custom_text_preprocessed )
        #predictions = model.predict(values)

        # Determine the result based on the prediction
        #binary_predictions = np.where(predictions >= 0.5, 1, 0)

# Display the predictions
    if predictions>0.47:
        result = "AI-GENERATED"
        
        return render_template('result.html', predictions=result,output=custom_text1)

    else: 
        result = "HUMAN"
        
        return render_template('result.html', predictions=result,output=custom_text1)

if __name__ == "__main__":
    app.run(debug=True)
    