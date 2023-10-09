# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding 

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Styles
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['patch.force_edgecolor'] = True

#Read the CSV file
df = pd.read_csv('spam_or_not_spam.csv')

#Show the number of missing(NAN,NaN,na) data for each column and drop them
df.isnull().sum()
df=df.dropna(subset=['email'])

#See what precentage of our data is spam/not-spam
df["label"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Not-Spam")
plt.legend(["Not-Spam", "Spam"])
plt.show()

#Seperate train and test data
df_train = df.sample(frac=.75, random_state=11)
df_test = df.drop(df_train.index)
print(df_train.shape, df_test.shape)

#Create y-data for analysis
y_train = df_train['label'].values
y_test = df_test['label'].values

#Create x-data for analysis
X_train = df_train['email'].values
X_test = df_test['email'].values

#Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_dict = tokenizer.index_word  #unique words

#Create sequences from sentences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

#Create pads with fix length
max_length = 40 #words per email
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
X_train_pad.shape

#Create Keras-model
vocab_size = len(word_dict) + 1
embeded_vector_size = 20

lstm_model = Sequential() #used for model composition
lstm_model.add(Embedding(input_dim=vocab_size, output_dim=embeded_vector_size, input_length=max_length)) # is used to provide a dense representation of words.
lstm_model.add(LSTM(2)) #used for creating LSTM layers in the network
lstm_model.add(Dense(1, activation='sigmoid')) #1 neuron sigmoid activation function

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.summary()

#Train model
history = lstm_model.fit(X_train_pad, y_train, 
                        epochs=50,
                        batch_size=64, 
                        validation_data=(X_test_pad, y_test))



#Test-Estimation

sms_test = ['Hi Paul, would you come around tonight']
sms_seq = tokenizer.texts_to_sequences(sms_test)

sms_pad = pad_sequences(sms_seq, maxlen=max_length, padding='post')
tokenizer.index_word
sms_pad
lstm_model.predict_classes(sms_pad)
#classified the text as no spam. Correct!


#Test-Estimation

sms_test = ['Free SMS service for anyone']
sms_seq = tokenizer.texts_to_sequences(sms_test)

sms_pad = pad_sequences(sms_seq, maxlen=max_length, padding='post')
tokenizer.index_word
sms_pad
lstm_model.predict_classes(sms_pad)
#classified the tet as spam. Correct again!



# evaluate the model
_, train_acc = lstm_model.evaluate(X_train_pad, y_train, verbose=0)
_, test_acc = lstm_model.evaluate(X_test_pad, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

#Compare results with expected test results
results = lstm_model.predict(X_test_pad, verbose=0)
results = results[:, 0]
plt.title('Compare results with expected test results')
plt.scatter(range(375),results,c='r')
plt.scatter(range(375),y_test,c='g')
plt.show()


# predict crisp classes for test set
yhat_classes = lstm_model.predict_classes(X_test_pad, verbose=0)
yhat_classes = yhat_classes[:, 0]

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)
