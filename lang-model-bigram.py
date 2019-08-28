## @Author Dhwani Serai
import sys
import math
from evaluator import metrics_sexism
from data_reader import file_classifier,char_tokenizer,word_tokenizer
from lang_model_unigram import UnigramLM

data_prefix = "../data/"
data_files = {"train": "amateur-aggregated-train.txt",
              "tune": "amateur-aggregated-tune.txt",
              "test": "amateur-aggregated-test.txt"}
class BigramLM:
    
    def __init__(self,preprocessing):
        self.flag = preprocessing
        self.bigramtokens = {}
        self.wordbigram = {} #unique bigrams for a word
        self.totallines = 0.0
        self.ksmoothing = 0.07
        self.k = 0.003
        self.beta = 50
        self.start = '_ '   #underscore then space
        self.join = '  ' #double space
        self.unigram = UnigramLM(False)

    def unigram_to_bigram(self,previous,current):
        return previous + self.join + current
                
    def vocabSize(self,word):   #unique bigram combination of word
        #it returns total words in a give model
        if word in self.wordbigram:
            return self.wordbigram[word]
        else:
            return 0
        #return 1000
        
    
    def update_tokens(self, tokens):
        #it updates the counts of words in unigram model
        self.totallines += 1
        tokens.insert(0,self.start)
        self.unigram.update_tokens(tokens)
        for i in range(1,len(tokens)):
            token_bi = self.unigram_to_bigram(tokens[i-1],tokens[i])
            self.wordbigram[tokens[i-1]] = 0;
            if not (token_bi in self.bigramtokens):
                self.bigramtokens[token_bi] = 0.0
                self.wordbigram[tokens[i-1]] += 1
            self.bigramtokens[token_bi] += 1

    def counts(self,previous,current):
        token_bi = self.unigram_to_bigram(previous,current) 
        if token_bi in self.bigramtokens: # and self.tokencounts[token] > 1:
            return self.bigramtokens[token_bi]
        else:
            return 0
                   
    def p(self, previous, current):
        #it gives the score of the token in the model

        if previous in self.wordbigram:
            self.ksmoothing =  self.wordbigram[previous]/(self.wordbigram[previous]+self.unigram.counts(previous))
            bigram_p = (self.counts(previous,current) + self.ksmoothing) /(self.ksmoothing *  self.vocabSize(previous) +
                                                                 self.unigram.counts(previous))
        else:
            bigram_p = 1/self.unigram.vocabSize()
##        trial probability
##        assert self.ksmoothing >= 0.0 and self.ksmoothing <= 1.0
##        p = bigram_p * (1-self.ksmoothing) + self.ksmoothing * self.unigram.p(current)
        p = bigram_p + self.beta*self.unigram.p(current)
        return p
    
    def logp(self,token_list):
        #it gives the total log probability of a tweet
        
        logprobab = 0.0
        token_list.insert(0,self.start)
        for i in range(1,len(token_list)):
            logprobab += math.log(self.p(token_list[i-1],token_list[i]))
        return logprobab

    def lmscore(self, token_list):
        return math.log(self.totallines) + self.logp(token_list)
            
    def train(self, tweets):
        #it takes list of tweet in given language model
        for token_list in tweets:
            self.update_tokens(token_list)                
    
def predictor(tweet,sexistmodel,nonsexistmodel, token_type):
    #compares language model of sexism and nonsexist and predicts after
    #comparison of total lm score
    if token_type == "char":
        token_list = char_tokenizer(tweet)
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
    segregated_tweets = file_classifier(file,preprocessor='none',token_type = 'char')
    lm_sexist = BigramLM(preprocessing=False)
    lm_sexist.train(segregated_tweets['sexism'])
    
    lm_nonsexist = BigramLM(preprocessing=False)
    lm_nonsexist.train(segregated_tweets['nonsexism'])
    
    print(metrics_sexism(predictor,'tune',lm_sexist,lm_nonsexist, token_type = 'char'))
    
