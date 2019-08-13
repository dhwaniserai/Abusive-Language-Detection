import sys
from evaluator import accuracy, metrics_sexism
import re

def mystrip(x):
    return x.strip()

def mysplit(tweet):
    tweet = tweet.replace('_', ' ')
    return tweet.split()

def sexism(splitted):
    if('feminazi' in splitted):
        if('laugh' in splitted):
            return False
        else:
            return True
    if('sexism' in splitted):
        return True
    if('dumb' in splitted and 'women' in splitted):
        return True
    if('assault' in splitted or 'harassment' in splitted or 'abuse' in splitted):
        if('sexual' in splitted or 'women' in splitted):
            return True
        else:
            return False
    if(('stop' in splitted or 'shut up' in splitted) and 'women in tech' in splitted):
        return True;
    if(('girls' in splitted or 'women' in splitted) and ('cringe' in splitted or 'cringing' in splitted)):
        return True
    if('#BlameOneNotAll' in splitted):
        return True
    return False
    
def racism(splitted):
    if('isis' in splitted or 'terrorism' in splitted or 'jihad' in splitted):
        return True
    if('blackmilk' in splitted):
        return True
    if('harassment' in splitted or 'harass' in splitted):
        return True
    if('abus(e|ing)' in splitted):
        if('child|children' in splitted):
            return False
        else:
            return True
    if('#killerblondes?' in splitted):
        return True
    if('cock' in splitted):
        return True
    if('dumb' in splitted and 'blondes?' in splitted):
        return True
    else:
        return False


def predict(tweet):
    tweet = preprocessor(tweet)
    prediction = ""
    splitted= mysplit(tweet)
    if (sexism(splitted)):
        prediction = "sexism"
    elif(racism(splitted)):
        prediction = "racism"
    else:
        prediction = "neither"
    return prediction

print(accuracy(predict,"tune"))
print(metrics_sexism(predict,"tune"))
sys.exit(0)

