# -*- coding : utf -8 -*-
"""
Created on Tue Jun 16 11:50:20 2020
@author : shruti yadav
"""
import numpy as np
import pandas as pd
import itertools
from sklearn . model_selection import train_test_split
from sklearn . feature_extraction . text import TfidfVectorizer
from sklearn . linear_model import PassiveAggressiveClassifier
from sklearn . metrics import accuracy_score , confusion_matrix
import emoji
import re
df=pd. read_csv (! train_covid .csv !,encoding =!ISO -8859 -1 !)
#Get shape and head
print (df. shape )
print (df. head ())
labels =df. target
print ( labels . head ())
fake_tweets = df[df. target == 0]
print ( fake_tweets . shape )
print ( fake_tweets . head (300) )
contractions = {
"ain !t": "am not",
" aren !t": "are not",
"can !t": " cannot ",
"can !t!ve": " cannot have ",
"!cause ": " because ",
" could !ve": " could have ",
" couldn !t": " could not",
" couldn !t!ve": " could not have ",
" didn !t": "did not",
" doesn !t": " does not",
"don !t": "do not",
" hadn !t": "had not",
" hadn !t!ve": "had not have ",
" hasn !t": "has not",
" haven !t": " have not",
"he !d": "he would ",
"he !d!ve": "he would have ",
"he !ll": "he will ",
"he !s": "he is",
"how !d": "how did",
"how !ll": "how will ",
"how !s": "how is",
"i!d": "i would ",
"i!ll": "i will ",
"i!m": "i am",
"i!ve": "i have ",
"isn !t": "is not",
"it !d": "it would ",
"it !ll": "it will ",
"it !s": "it is",
"let !s": "let us",
"ma !am": " madam ",
" mayn !t": "may not",
" might !ve": " might have ",
" mightn !t": " might not",
" must !ve": " must have ",
" mustn !t": " must not",
" needn !t": " need not",
" oughtn !t": " ought not",
" shan !t": " shall not",
"sha !n!t": " shall not",
"she !d": "she would ",
"she !ll": "she will ",
"she !s": "she is",
" should !ve": " should have ",
" shouldn !t": " should not",
" that !d": " that would ",
" that !s": " that is",
" there !d": " there had",
" there !s": " there is",
" they !d": " they would ",
" they !ll": " they will ",
" they !re": " they are",
" they !ve": " they have ",
" wasn !t": "was not",
"we !d": "we would ",
"we !ll": "we will ",
"we !re": "we are",
"we !ve": "we have ",
" weren !t": " were not",
" what !ll": " what will ",
" what !re": " what are",
" what !s": " what is",
" what !ve": " what have ",
" where !d": " where did",
" where !s": " where is",
"who !ll": "who will ",
"who !s": "who is",
"won !t": " will not",
14" wouldn !t": " would not",
"you !d": "you would ",
"you !ll": "you will ",
"you !re": "you are",
"thx" : " thanks "
}
def remove_contractions ( text ):
return contractions [ text . lower ()] if text . lower () in
contractions . keys () else text
df[!text !]= df[!text !]. apply ( remove_contractions )
# print (df. head ())
def clean_dataset ( text ):
# Remove hashtag while keeping hashtag text
text = re.sub(r!#!,!!, text )
# Remove HTML special entities (e.g. &amp ;)
text = re.sub(r!\&\w*; !, !!, text )
# Remove tickers
text = re.sub(r!\$\w*!, !!, text )
# Remove hyperlinks
text = re.sub(r!https ?:\/\/.*\/\ w*!, !!, text )
# Remove whitespace ( including new line characters )
text = re.sub(r!\s\s+!,!!, text )
text = re.sub(r![ ]{2 , }!,! !,text )
# Remove URL , RT , mention (@)
text = re.sub(r!http (\S)+!, !!,text )
text = re.sub(r!http ... !, !!,text )
text = re.sub(r!(RT|rt)[ ]*@[ ]*[\ S]+ !,!!,text )
text = re.sub(r!RT[ ]?@!,!!,text )
text = re.sub(r!@[\S]+ !,!!,text )
# Remove words with 4 or fewer letters
text = re.sub(r!\b\w{1 ,4}\b!, !!, text )
#&, < and >
text = re.sub(r!&amp ;? !, !and !,text )
text = re.sub(r!&lt;!,!<!,text )
text = re.sub(r!&gt;!,!>!,text )
# Remove characters beyond Basic Multilingual Plane (BMP) of Unicode :
text = !!. join (c for c in text if c <= !\ uFFFF !)
text = text . strip ()
# Remove misspelling words
text = !!. join (!!. join (s) [:2] for _, s in itertools . groupby ( text ))
# Remove emoji
text = emoji . demojize ( text )
text = text . replace (":"," ")
text = ! !. join ( text . split ())
text = re.sub(" ([^\ x00 -\ x7F ])+"," ",text )
# Remove Mojibake ( also extra spaces )
text = !
!. join (re.sub("[^\ u4e00 -\ u9fa5 \u0030 -\ u0039 \u0041 -\ u005a \u0061 -\ u007a ]",
" ", text ). split ())
return text
df[!text !] =df[!text !]. apply ( clean_dataset )
print (df. head ())
print (df. shape )
x_train ,x_test , y_train , y_test = train_test_split (df[!text !], labels ,
test_size =0.2 , random_state =7)
15# DataFlair - Initialize a TfidfVectorizer
from sklearn . feature_extraction . text import CountVectorizer
vectorizer = CountVectorizer ()
vectorizer .fit( x_train )
X_train = vectorizer . transform ( x_train )
X_test = vectorizer . transform ( x_test )
from sklearn . naive_bayes import MultinomialNB
clf = MultinomialNB ().fit( X_train , y_train )
predicted = clf. predict ( X_test )
print (np. mean ( predicted == y_test ))
from sklearn . linear_model import SGDClassifier
clf = SGDClassifier ().fit( X_train , y_train )
predicted = clf. predict ( X_test )
print (np. mean ( predicted == y_test ))
from sklearn . model_selection import GridSearchCV
from sklearn . model_selection import ParameterGrid
import parfit . parfit as pf
from sklearn . metrics import roc_auc_score
param_grid = {
!alpha !: [1e -4, 1e -3, 1e -2, 1e -1, 1e0 , 1e1 , 1e2 , 1e3], # learning rate
!loss !: [!log !,!hinge !], # logistic regression ,
!penalty !: [!l2 !],
!n_jobs !: [ -1]
}
paramGrid = ParameterGrid ( param_grid )
bestModel , bestScore , allModels , allScores = pf. bestFit ( SGDClassifier ,
paramGrid ,
X_train , y_train , X_test , y_test ,
metric = roc_auc_score ,
scoreLabel = "AUC")
print ( bestModel , bestScore )
clf = SGDClassifier ( alpha =0.001 , average =False , class_weight =None ,
early_stopping =False , epsilon =0.1 , eta0 =0.0 ,
fit_intercept =True ,
l1_ratio =0.15 , learning_rate =!optimal !, loss =!log !,
max_iter =1000 ,
n_iter_no_change =5, n_jobs =-1, penalty =!l2 !, power_t =0.5 ,
random_state =None , shuffle =True , tol =0.001 ,
validation_fraction =0.1 , verbose =0,
warm_start = False ).fit( X_train , y_train )
data = {!text !:[ !Last evening , I interacted with various NGO
representatives on # AatmanirbharBharat and role of NGOs , a unique
initiative by MahaNGO Federation . Stressed on need to focus more on
opportunities for India post # COVID19 pandemic .!,!After the bill is
passed into law in Provincial Assembly Sindh - video link conference is
introduced and is being demonstrated for the first time by Honourable
Chief Minister and Worthy Speaker Sindh Assembly in picture below .
# govtsindh !]}
test_df =pd. DataFrame ( data )
test_df [!text !]= test_df [!text !]. apply ( remove_contractions )
16## Clean Data set
test_df [!text !] = test_df [!text !]. apply ( clean_dataset )
X_test = vectorizer . transform ( test_df [!text !])
y_pred =clf. predict ( X_test )
print ( y_pred )
