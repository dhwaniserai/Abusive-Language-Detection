import sys
import math
from evaluator import accuracy

data_prefix = "../data/"
data_files = {"train": "amateur-aggregated-train.txt",
              "tune": "amateur-aggregated-tune.txt",
              "test": "amateur-aggregated-test.txt"}
unigram_counts_sexism = {}
unigram_counts_nonsexist = {}
unigram_counts_nonsexist["__TOTAL__"] = 0.0
unigram_counts_sexism["__TOTAL__"] = 0.0

def mystrip(x):
    return x.strip()

def mysplit(tweet):
    tweet = tweet.replace('_', ' ')
    tweet.lower()
    return tweet.split()

def train_sexism(tweet):
    splitted_tweet=tweet.split()
    unigram_counts_sexism["__TOTAL__"] += 1.0
    for word in splitted_tweet:
        if not (word in unigram_counts_sexism):
            unigram_counts_sexism[word] = 1
        else:
            unigram_counts_sexism[word] += 1
            
def train_nonsexist(tweet):
    splitted_tweet=tweet.split()
    unigram_counts_nonsexist["__TOTAL__"] += 1
    for word in splitted_tweet:
        if not (word in unigram_counts_nonsexist):
            unigram_counts_nonsexist[word] = 1.0
        else:
            unigram_counts_nonsexist[word] += 1.0
                
def lm_sexism(tweet):
    probab=0.0
    splitted_tweet=tweet.split()
    for word in splitted_tweet:
        if word in unigram_counts_sexism:
            probab += math.log(unigram_counts_sexism[word])
    return probab        
def lm_nonsexist(tweet):
    probab=0.0
    splitted_tweet=tweet.split()
    for word in splitted_tweet:
        if word in unigram_counts_nonsexist:
            probab += math.log(unigram_counts_nonsexist[word])
    return probab
    


def test_model(mode):
    total_line_count = unigram_counts_sexism["__TOTAL__"] + unigram_counts_nonsexist["__TOTAL__"]
    prior_sexist = unigram_counts_sexism["__TOTAL__"]/total_line_count
    prior_nonsexist = unigram_counts_nonsexist["__TOTAL__"]/total_line_count
    filename = data_prefix + data_files[mode]
    correct_lines = 0
    total_lines = 0
    for line in open(filename):
        line = line.strip()
        splitted = line.split()
        label = splitted[0]
        tweet = line[len(label)+1:]
        probab_sexism = lm_sexism(tweet)
        probab_nonsexist = lm_nonsexist(tweet)  #non-normalized
        if (probab_sexism*prior_sexist) > (probab_nonsexist*prior_nonsexist):
            testlabel = 'sexism'
        else:
            testlabel = 'nonsexist'
        if testlabel=='sexism' and label=='sexism':
            correct_lines += 1
        elif testlabel=='nonsexist' and not label=='sexism':
            correct_lines += 1
        total_lines += 1
    accuracy_value = correct_lines * 1.0 / total_lines
    return (accuracy_value)
        
def train_model(mode):
    filename = data_prefix + data_files[mode]
    total_line_count = 0
    for line in open(filename):
        total_line_count += 1
        line = line.strip()
        splitted = line.split()
        label = splitted[0]
        tweet = line[len(label)+1:]
        if label=='sexism':
            train_sexism(tweet)
        else:
            train_nonsexist(tweet)
    for word in unigram_counts_nonsexist:
        unigram_counts_nonsexist[word] /= unigram_counts_nonsexist["__TOTAL__"]
    for word in unigram_counts_sexism:
        unigram_counts_sexism[word] /= unigram_counts_sexism["__TOTAL__"]

if __name__ == '__main__':
    #train_model('train')
    train_model('tune')
    accuracy = test_model('test')
    print(accuracy)
            
    
