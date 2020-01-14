import nltk
from newspaper import  Article
import  random
import  string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

from tkinter import *

warnings.filterwarnings('ignore')
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)

article = Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521')
article.download()
article.parse()
article.nlp()
corpus = article.text
#print(corpus)
#just completed ggetting the data from the website like that

#tookenization
text = corpus
sent_tokens = nltk.sent_tokenize(text) # converting the text into list of sentences
#print(sent_tokens) OUR DATABASE WE WILL KEEP ON ADDING ONTO IT DURING CONVERSATION TO MAKE IT LAARGE AND BIG AND BETTER AND SMART


#creatin dict to remove punctucations
remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)
#print(string.punctuation)
# now i have the dict of punct with non values
#print(remove_punct_dict)

# now we will lemmatize words(meaning only returning the dictoray word )
#that has some emnaing for example whats up = what, up

def LemNormalize(text):
    # this fucntion will return lower case words  and translate will remove
    # all the items present in the remove punct dict whicha are basicallt a punctuations
    return  nltk.word_tokenize(text.lower().translate(remove_punct_dict))

# printing the tokenizwed text received from the fucntion
#print(LemNormalize(text)) # this is splited word by word where as in line 25 it was \n and lines
# so now i have the text from the website into the list of the wordss as tokens

#Keyword matching and greeting inpust
GREETING_INPUTS = ["hi","hello","ola","greetings","wassup","sup","hi nice to meet you","Its Pleasure to meet you"]
#greetings responses to the user fun part
GREETING_RESPONSES = ["ey yo!","howdy","hey","what's good fam","hello","hey there!"]
ROBO_OFF_TOPIC_RESPONSES = ["Sorry fam, stay on topic so I can help you","I aologize ,Idon't understand that"]
ROBO_LATER = ["DOCBOT: Aii Later homes","DOCBOT: Chat with you later !","DOCBOT: Eat Sleep Stay Healthy Peace !","Later fam !","See you soon!","Nice chatting see you soon!","I'M OUT!","Stay Healthy drink water for your kidneys!"]
ROBO_WELCOME = ["DOCBOT: You are Welcome!","My RAM feels so happy to help you","Welcome!","You got it.","Don’t mention it,I am machine made for help that makes my Operating System happy.","No worries I'm always here to help you about this.","Not a problem",
                "My pleasure","I'm happy to help","Not at all ,I'm here to help you","Anytime, and by anytime I mean anytime you turn your laptop and run DOCBOT"]


# function to throw greeting to user
def greeting(sentence):
    # if the user input is a greeting then return randomly chosen greeting response
    # sentence is the user input
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
##GENERATES THE RESPINSE
def response(user_response):


# The user response / query
# user_response = 'what is chronic kidney disease'
    user_response = user_response.lower()  # makes lower case

# chatbot resonses
    robo_response = ''

# append the user response to the sentemce list
# KEEPS ON ADDING STUFF TO THE SO CALLED DATABSDE TO MAKE IT SMART
    sent_tokens.append(user_response)
# tfidfvectorizer objcet
# i have a vector of most used word and rarely used
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
# TfidfVectorizer - Transforms text to feature vectors that can be used as input to estimator.


# convert text to matrix of TF-IDF features
    tfidf = TfidVec.fit_transform(sent_tokens)

##printing thr tfidf features

# get the measure of the similarity
# comparing the user user_response  vs everything in the text(on doc)
    vals = cosine_similarity(tfidf[-1], tfidf)  # this will calculate the similarity between user repsonse an deverything

# print the simalrity scores betwwn 0 and 1 of everything

# get the index of the most similar sentence/text  to the user question/resonse
# the -1 is the user response itksef we ewant the next mo
    idx = vals.argsort()[0][-2]  # index of the most similar text in the doc
# vals is the text in numbers of similarity
# reduce the dimensionality of vals list of list  to one list
    flat = vals.flatten()

# sort the list in ascending order
    flat.sort()

# get the most similar score to the user response
    score = flat[-2]  # score will give the highest similarty between user response and the doc text we have
# similaruity score
# closert to 1 higheer the similarity
# if the variable score is 0 then there is no text similar to userresponse
    if (score == 0):
        robo_response = robo_response + random.choice(ROBO_OFF_TOPIC_RESPONSES)

    else:
    # we have a response to the user input we can RESPOND BACK HARD !!
        robo_response = robo_response + sent_tokens[idx]
    #    idx is the most similar response from the text file we have

# print the chatbot response

    return robo_response


flag = True
print("DOCBot : Hey sup!, I am Doctor Bot, they call me DOCBOT and you can ask me anything about Chronic Kidney Disease. If you want to exit type Bye!")

while (flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print(random.choice(ROBO_WELCOME))
        else:
            if(greeting(user_response) != None):
                print("DOCBOT: "+greeting(user_response))
            else:
                print("DOCBOT: "+response(user_response))
    else:
        print(random.choice(ROBO_LATER))


