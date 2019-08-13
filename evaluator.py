import sys
#from data_reader import file_classifier

data_prefix = "../data/"
data_files = {"train": "amateur-aggregated-train.txt",
              "tune": "amateur-aggregated-tune.txt",
              "test": "amateur-aggregated-test.txt"}


def accuracy(predictor, mode):
    if not (mode in data_files):
        raise KeyError()
    total_lines = 0
    correct_lines = 0
    filename = data_prefix + data_files[mode]
    for line in open(filename):
        line = preprocessor(line)
        splitted = line.split()
        label = splitted[0]
        tweet = line[len(label)+1:]
        prediction = predictor(tweet)
        if prediction == label:
            correct_lines += 1
        total_lines += 1
    accuracy_value = correct_lines * 1.0 / total_lines
    return (accuracy_value)

#metrics={"accuracy","recall","precision","f1score"}
def metrics_sexism(predictor, mode,sexist_obj,nonsexist_obj,token_type):
    if not (mode in data_files):
        raise KeyError()
    metrics={}
    conf_arr = [[0,0],[0,0]]    #confusion matrix
    total_lines = 0
    correct_lines = 0
    filename = data_prefix + data_files[mode]
    for line in open(filename):
        line = line.strip()
        splitted = line.split()
        label = splitted[0]
        tweet = line[len(label)+1:]
        prediction = predictor(tweet, sexist_obj,nonsexist_obj, token_type)
        if label == "sexism":
            if label == prediction:
                conf_arr[1][1] +=1
                correct_lines += 1
            else:
                conf_arr[1][0] += 1
        else:
            if prediction != "sexism":
                correct_lines +=1
                conf_arr[0][0] += 1
            else:
                conf_arr[0][1] += 1
        total_lines += 1
    metrics['accuracy']=correct_lines * 1.0 / total_lines
    metrics['precision']=conf_arr[1][1]/(conf_arr[1][1]+conf_arr[0][1])
    metrics['recall']=conf_arr[1][1]/(conf_arr[1][1]+conf_arr[1][0])
    metrics['f1score']=2*metrics['precision']*metrics['recall']/(metrics['precision']+metrics['recall'])
    print('   no\tyes')
    print('no '+str(conf_arr[0][0]) +'\t'+ str(conf_arr[0][1]))
    print('yes '+str(conf_arr[1][0]) +'\t'+ str(conf_arr[1][1]))
    return (metrics)
    

def accuracy_sexism(predictor, mode):
    if not (mode in data_files):
        #raise TypeError("Mode is invalid")
        raise KeyError()
    total_lines = 0
    correct_lines = 0
    filename = data_prefix + data_files[mode]
    for line in open(filename):
        line = line.strip()
        splitted = line.split()
        label = splitted[0]
        tweet = line[len(label)+1:]
        prediction = predictor(tweet)
        if label == "sexism":
            if label == prediction:
                correct_lines += 1
        else:
            if prediction != "sexism":
                correct_lines +=1
        total_lines += 1
    accuracy_value = correct_lines * 1.0 / total_lines
    return (accuracy_value)

if __name__ == '__main__':
    print(data_files["train"])
    print(data_files["tune"])
    print(data_files["test"])
    
