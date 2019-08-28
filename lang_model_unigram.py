## @Author Dhwani Serai
import sys
import math
from data_reader import file_classifier,char_tokenizer,word_tokenizer
from evaluator import metrics_sexism

class UnigramLM:
    # this is unigram language model

    def __init__(self,preprocessing):
        self.flag=preprocessing
        self.uniquetokens = {}
        self.totallines = 0.0
        self.totaltokens = 0.0  #total occurences of words
        self.ksmoothing = 0.2

    def vocabSize(self):
        #it returns total words in a give model
        return len(self.uniquetokens)
        #return 1000
        
    
    def update_tokens(self,tweet):
        #it updates the counts of words in unigram model
        self.totallines += 1 
        for token in tweet:
            self.totaltokens += 1.0
            if not (token in self.uniquetokens):
                self.uniquetokens[token] = 0.0
            self.uniquetokens[token] += 1

    def counts(self,token):
        if token in self.uniquetokens: # and self.tokencounts[token] > 1:
            return self.uniquetokens[token]
        else:
            return 0
                   
    def p(self, token):
        #it gives the score of the token in the model
        p = (self.counts(token) + self.ksmoothing) / (self.totaltokens + self.ksmoothing * self.vocabSize())
        return p
    
    def logp(self,tweet):
        #it gives the total log probability of a tweet
        logprobab=0.0
        for token in tweet:
            logprobab += math.log(self.p(token))
        return logprobab

    def lmscore(self, tweet):
        return math.log(self.totallines) + self.logp(tweet)
            
    def train(self, tweets):
        #it takes list of tweet in given language model
        for token_list in tweets:
            self.update_tokens(token_list)
            
def predictor(tweet,sexistmodel,nonsexistmodel, token_type):
    #compares language model of sexism and nonsexist and predicts after
    #comparison of total lm score
    if token_type=="char":
            token_list=char_tokenizer(tweet)
    else:
        token_list=word_tokenizer(tweet)
    probab1 = sexistmodel.lmscore(token_list)
    probab2 = nonsexistmodel.lmscore(token_list)
    if probab1 > probab2 :
        return 'sexism'
    else:
        return 'nonsexism'

if __name__ == '__main__':
    file = "../data/amateur-aggregated-train.txt"
    segregated_tweets=file_classifier(file,preprocessor='none')
    lm_sexist = UnigramLM(preprocessing=False)
    lm_sexist.train(segregated_tweets['sexism'])
    
    lm_nonsexist = UnigramLM(preprocessing=False)
    lm_nonsexist.train(segregated_tweets['nonsexism'])
  
    print(metrics_sexism(predictor,'tune',lm_sexist,lm_nonsexist, token_type = 'char'))
            
    
