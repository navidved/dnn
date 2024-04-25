# dataset : https://github.com/OmkarPathak/Playing-with-datasets/blob/master/Email%20Spam%20Filtering/emails.csv

# Importing necessary libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')
 
# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
 
import warnings
warnings.filterwarnings('ignore')


# Loading Dataset
data = pd.read_csv('classifications/ds/emails.csv')
print(data.head())

# check how many such tweets data we have let’s print the shape of the data frame.
data.shape

# For a better understanding, we’ll plot these counts:
sns.countplot(x='spam', data=data)
plt.show()

# We can clearly see that number of samples of Ham is much more than that of Spam which implies that the dataset we are using is imbalanced. 
# Downsampling to balance the dataset
ham_msg = data[data.spam == 0]
spam_msg = data[data.spam == 1]
ham_msg = ham_msg.sample(n=len(spam_msg), random_state=42)
 
# Plotting the counts of down sampled dataset
balanced_data = pd.concat([ham_msg, spam_msg],ignore_index=True).reset_index(drop=True)
plt.figure(figsize=(8, 6))
sns.countplot(data = balanced_data, x='spam')
plt.title('Distribution of Ham and Spam email messages after downsampling')
plt.xlabel('Message types')

# Text Preprocessing
# Textual data is highly unstructured and need attention in many aspects:
# Stopwords Removal
# Punctuations Removal
# Stemming or Lemmatization
# Although removing data means loss of information we need to do this to make the data perfect to feed into a machine learning model.

balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')
print(balanced_data.head())


punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)
 
balanced_data['text']= balanced_data['text'].apply(lambda x: remove_punctuations(x))
print(balanced_data.head())

# The below function is a helper function that will help us to remove the stop words.

def remove_stopwords(text):
    stop_words = stopwords.words('english')
 
    imp_words = []
 
    # Storing the important words
    for word in str(text).split():
        word = word.lower()
 
        if word not in stop_words:
            imp_words.append(word)
 
    output = " ".join(imp_words)
 
    return output
 
 
balanced_data['text'] = balanced_data['text'].apply(lambda text: remove_stopwords(text))
print(balanced_data.head())

# A word cloud is a text visualization tool that help’s us to get insights into the most frequent words present in the corpus of the data.
def plot_word_cloud(data, typ):
    email_corpus = " ".join(data['text'])
 
    plt.figure(figsize=(7, 7))
 
    wc = WordCloud(background_color='black',
                   max_words=100,
                   width=800,
                   height=400,
                   collocations=False).generate(email_corpus)
 
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {typ} emails', fontsize=15)
    plt.axis('off')
    plt.show()
 
plot_word_cloud(balanced_data[balanced_data['spam'] == 0], typ='Non-Spam')
plot_word_cloud(balanced_data[balanced_data['spam'] == 1], typ='Spam')

# Word2Vec Conversion
# We cannot feed words to a machine learning model because they work on numbers only.
# So, first, we will convert our words to vectors with the token IDs to the corresponding 
# words and after padding them our textual data will arrive to a stage where we can feed it to a model.

#train test split
train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['text'],
                                                    balanced_data['spam'],
                                                    test_size = 0.2,
                                                    random_state = 42)

# We have fitted the tokenizer on our training data we will use it to convert the training and validation data both to vectors.

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
 
# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)
 
# Pad sequences to have the same length
max_len = 100  # maximum sequence length
train_sequences = pad_sequences(train_sequences,
                                maxlen=max_len, 
                                padding='post', 
                                truncating='post')
test_sequences = pad_sequences(test_sequences, 
                               maxlen=max_len, 
                               padding='post', 
                               truncating='post')


# Model Development and Evaluation
# We will implement a Sequential model which will contain the following parts:

# Three Embedding Layers to learn featured vector representations of the input vectors.
# An LSTM layer to identify useful patterns in the sequence.
# Then we will have one fully connected layer.
# The final layer is the output layer which outputs probabilities for the two classes. 


# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                    output_dim=32, 
                                    input_length=max_len))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
 
# Print the model summary
model.summary()

# While compiling a model we provide these three essential parameters:

# optimizer – This is the method that helps to optimize the cost function by using gradient descent.
# loss – The loss function by which we monitor whether the model is improving with training or not.
# metrics – This helps to evaluate the model by predicting the training and the validation data.

model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics = ['accuracy'],
              optimizer = 'adam')

# Callbacks are used to check whether the model is improving with each epoch or not. 
# If not then what are the necessary steps to be taken like ReduceLROnPlateau decreases
# the learning rate further? Even then if model performance is not improving then training 
# will be stopped by EarlyStopping. We can also define some custom callbacks to stop
# training in between if the desired results have been obtained early.


es = EarlyStopping(patience=3,
                   monitor = 'val_accuracy',
                   restore_best_weights = True)
 
lr = ReduceLROnPlateau(patience = 2,
                       monitor = 'val_loss',
                       factor = 0.5,
                       verbose = 0)


# Let us now train the model:
# Train the model
history = model.fit(train_sequences, train_Y,
                    validation_data=(test_sequences, test_Y),
                    epochs=20, 
                    batch_size=32,
                    callbacks = [lr, es]
                   )

# Now, let’s evaluate the model on the validation data.

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('Test Loss :',test_loss)
print('Test Accuracy :',test_accuracy)

# Thus, the training accuracy turns out to be 97.44% which is quite satisfactory.

# Model Evaluation Results
# Having trained our model, we can plot a graph depicting the variance of training and validation accuracies with the no. of epochs.


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
pass









