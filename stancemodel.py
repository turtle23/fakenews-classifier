# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:27:23 2019

@author: Sneha
"""

import pandas as pd
import numpy as np
from datetime import datetime
from nltk.corpus import stopwords
import time
import io
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.layers import Dense ,  Flatten , Dropout ,Concatenate 
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

from sklearn.metrics import confusion_matrix
import pickle

start_time = time.time()
datestring = datetime.strftime(datetime.now() , '%Y-%m-%d_%H-%M-%S')
f = open('lstm1'+ datestring + '.txt' , 'w')
print(f.name)
#load embedding vectors
f.write('Loading embedding vectors...')
fname ='wiki-news-300d-1M.vec/wiki-news-300d-1M.vec'
fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#n, d = map(int, fin.readline().split())

embedding_index = {}
for line in fin:
        tokens = line.rstrip().split(' ')
        embedding_index[tokens[0]] = np.asarray(tokens[1:11] , dtype = 'float32')
f.write('total word vectors' + str(len(embedding_index)) + '\n')
print('total word vectors' , len(embedding_index))
print('embedding index of fire',embedding_index['fire'][0:10] )
f.write('embedding index of fire' + str(embedding_index['fire'][0:10]) + '\n')


def preprocessing(df ,column_name):
    f.write('Inside preprocessing' + '\n')
    #Lowercase 
    df[column_name] = df[column_name].apply(lambda x : " ".join(x.lower() for x in x.split()))
    f.write('lowering the input : '+ df[column_name][0] + '\n')
    #Remove Punctuation
    df[column_name] = df[column_name].str.replace('[^\w\s]' , '')
    f.write('after removing punct : '+ df[column_name][0] + '\n')
    #Remove stopwords
    stopword = stopwords.words('english')
    df[column_name] = df[column_name].apply(lambda x: " ".join(x for x in x.split() if x not in stopword))
    f.write('after removing stopwords : '+ df[column_name][0] + '\n')
    
    return df

#Tokenization
def tokenization(df , column_name ):
    f.write('Inside tokenization' + '\n')
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(df[column_name])
    
    return tokenizer 

#save tokenizer pickle file
def load_or_savetokenizer(filename , df ,column_name ):
        try:
            with open(filename , "rb") as f:
                tokenizer = pickle.load(f)
                #return foo
        except Exception as e:
            print(e)
            tokenizer = tokenization(df , column_name )
            with open(filename , 'wb') as f:
                pickle.dump(tokenizer ,f)
                print(f'{filename} saved')
                
        return tokenizer
 
def create_embeddingmatrix(embedding_index , vocabsize ,embed_dim ,word_index):
    f.write('Inside create_embeddingmatrix' + '\n')
    f.write('creating embedding matrix for dataset...' + '\n')
    words_not_found = []
    embedding_matrix = np.zeros((vocabsize , embed_dim))
    for word ,  i in word_index.items():
        if i >= vocabsize:
            continue
        embeddingvector = embedding_index.get(word)
        if (embeddingvector is not None) and len(embeddingvector) > 0:
            #words not found in embedding_index will beall zeros.
            embedding_matrix[i] = embeddingvector
            
        else:
            words_not_found.append(word)
    print('embedding matrix shape-- :' ,embedding_matrix.shape)
    nullword_count = np.sum(np.sum(embedding_matrix ,axis = 1) == 0)
    print('total null word embeddings:' , nullword_count)
    f.write('total null word embeddings:' + str(nullword_count) + '\n')
    
    return embedding_matrix , words_not_found           

#create model 
def model_lstm(leftvocabsize ,leftembed_dim ,leftembedding_matrix ,bodymaxlen ,
               rightvocabsize ,rightembed_dim , rightembedding_matrix ,headlinemaxlen):
    f.write('Inside model_lstm' + '\n')
    #LSTM arcitecture for 'article body'
    f.write('training LSTM architecture for articlebody...')
    leftbranch = Sequential()
    leftbranch.add(Embedding(leftvocabsize ,leftembed_dim ,
            weights = [leftembedding_matrix] ,input_length = bodymaxlen ,trainable = False)
            )
    leftbranch.add(Dropout(0.2))
    leftbranch.add(LSTM(80 , return_sequences= True))
    leftbranch.add(Dropout(0.2))
    leftbranch.add(Flatten())
    
    #LSTM architecturen for headline
    rightbranch = Sequential()
    rightbranch.add(Embedding(rightvocabsize ,rightembed_dim ,
            weights = [rightembedding_matrix] ,input_length = headlinemaxlen ,trainable = False)
            )
    rightbranch.add(Dropout(0.2))
    rightbranch.add(LSTM(10 , return_sequences= True))
    rightbranch.add(Dropout(0.2))
    rightbranch.add(Flatten())
    
    #merge the output tensors of 2 models (functional layer)
    conc = Concatenate()([leftbranch.output , rightbranch.output])
    out = Dense(4 , activation= 'softmax')(conc)
    modelall = Model([leftbranch.input , rightbranch.input] , out)
    opt = Adam(lr = 0.001)
    modelall.compile(optimizer = opt ,
                     loss = 'categorical_crossentropy',
                     metrics = ['accuracy'])
    return modelall



#read csv files
f.write('read csv files' + '\n')
train_body_df = pd.read_csv('fnc-1-master/train_bodies.csv' ,encoding = 'utf-8')
train_stances_df = pd.read_csv('fnc-1-master/train_stances.csv' ,encoding = 'utf-8')

f.write('train_body_df.shape : ' +str(train_body_df.shape) + '\n')
f.write('train_stances_df.shape : ' +str(train_stances_df.shape) + '\n')

#preprocessing
f.write('preprocessing' + '\n')
train_body_df = preprocessing(train_body_df ,'articleBody')
print(train_body_df['articleBody'][0])
train_stances_df = preprocessing(train_stances_df ,'Headline')
print(train_stances_df['Headline'][0])
print('total time(in seconds) for preprocessing ..' ,time.time() - start_time)

#merge two dataframes
df = train_body_df.merge(train_stances_df ,how= 'left')
print(df.head())
f.write('dataframe shape after merging' + str(df.shape) + '\n')
#check total null values
percent = (df.isnull().sum()).sort_values(ascending = False)
f.write('null values...' +str(percent) + '\n')
print(f'percent of null value:' ,percent)
#df = df.dropna(axis = 0)

#split train and test set
df_train ,df_test = train_test_split(df , test_size = 0.05 ,random_state = 43)
print(f'df_train size = {df_train.shape} , df_test_size = {df_test.shape}')
f.write('train and test shape' + str(df_train.shape ) +' ' + str(df_test.shape)+ '\n')

#calculate maximum lngth of a sentence 
maxlen = df_train['articleBody'].str.split().str.len()
bodymaxlen = max(maxlen)

maxlen_ = df_train['Headline'].str.split().str.len()
headlinemaxlen = max(maxlen_)
print(f'body and headline max length : {bodymaxlen:5}, {headlinemaxlen} ')
    
#tokenize article body and headline
s_time  = time.time()
left_tokenizer  = load_or_savetokenizer('left_tokenizer' , df_train , 'articleBody' )
Xtrain_left = left_tokenizer.texts_to_sequences(df_train['articleBody'])
print(Xtrain_left[0])
#padding 
Xtrain_left = pad_sequences(Xtrain_left ,padding = 'post' ,maxlen = bodymaxlen)
print(Xtrain_left[0])

right_tokenizer  = load_or_savetokenizer('right_tokenizer' , df_train , 'Headline' )   
Xtrain_right = right_tokenizer.texts_to_sequences(df_train['Headline'])
print(Xtrain_left[0])
#padding 
Xtrain_right = pad_sequences(Xtrain_right ,padding = 'post' ,maxlen = headlinemaxlen)
print(Xtrain_right[0])
print('total time for tokenization(seconds)' , time.time() - s_time)

wordindex_left = left_tokenizer.word_index
wordindex_right = right_tokenizer.word_index

leftvocabsize = len(wordindex_left) + 1
f.write('left Vocab_size : ' +str(leftvocabsize) + '\n')
print('left Vocab_size :' ,leftvocabsize)
rightvocabsize = len(wordindex_right) + 1
f.write('right Vocab_size : ' +str(rightvocabsize) + '\n')
print('right Vocab_size :' ,rightvocabsize)


#calling create_embeddingmatrix function
leftembed_dim = 10
rightembed_dim = 10
leftembedding_matrix ,leftwords_notfound = create_embeddingmatrix(embedding_index ,leftvocabsize ,
                                            leftembed_dim , wordindex_left)
rightembedding_matrix ,rightwords_notfound = create_embeddingmatrix(embedding_index ,rightvocabsize ,
                                            rightembed_dim , wordindex_right)

f.write('leftembedding_matrix size' +str(leftembedding_matrix.shape) + '\n')
f.write('rightembedding_matrix size' +str(rightembedding_matrix.shape) + '\n')

#label encoding for class values as integers
encoder = LabelEncoder()
labels = ['agree' ,'disagree' , 'discuss' ,'unrelated']
encoder.fit(labels)
encoded_Y = encoder.transform(df_train['Stance'])
#convert integers to dummy variables (one hot encodings)
from keras.utils import np_utils
y_train = np_utils.to_categorical(encoded_Y)



f.write('calling model_lstm...')
model = model_lstm(leftvocabsize ,leftembed_dim ,leftembedding_matrix ,bodymaxlen ,
               rightvocabsize ,rightembed_dim , rightembedding_matrix ,headlinemaxlen)
history = model.fit([Xtrain_left ,Xtrain_right] ,y_train, batch_size = 100,
                    epochs = 15 ,validation_split = 0.2,
                    callbacks = [EarlyStopping(monitor = 'val_loss' ,min_delta = 0.0001 ,patience =5)])

#save model 
model.save('stancemodel.h5')  # creates a HDF5 file 'my_model.h5'

xtestbody = left_tokenizer.texts_to_sequences(df_test['articleBody'])
xtestbody = pad_sequences(xtestbody , padding = 'post' ,maxlen = bodymaxlen)

xtest_headline = right_tokenizer.texts_to_sequences(df_test['Headline'])
xtest_headline = pad_sequences(xtest_headline ,padding = 'post' ,maxlen= headlinemaxlen)

prediction = model.predict([xtestbody , xtest_headline])
predicted_labels = prediction.argmax(axis = -1)
y_test = encoder.transform(df_test['Stance'])
output = (y_test == predicted_labels)
x = output*1
total_correct = x.sum()
print(f'total correct {total_correct}' )
f.write('total correct predictions' + str(total_correct) + '\n')
f.write('toal time taken (in sec)' + str(time.time() - start_time) + '\n')
f.close()

