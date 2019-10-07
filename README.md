# Abusive Language Detection (ALD)

<b> What is this software? </b><br>
  This software helps in identifying hateful tweets. It can be used to report cyber-bullying.
  This software currently identifies sexist and racist tweets.

<b>How to use ALD ? </b><br>
It trains itself by reading a file containing a tweet label and the tweet.
On the basis of that trained model then it predicts the label for a tweet
<br><br>Note: This test file also contains the label which is used to find the accuracy of the model<br>
<b>Current Accuracy of Model: 0.857 . Precision = 0.54  Recall = 0.67 . </b> on test datasets
<br> The above results can be verified by running evaluator.py
<br> lang-model.py is a rule based language model based on general trends
<br> lang_model_unigram is a unigram language model
<br> lang-model-bigram is a bigram language model
<b> Requirements : Python3</b>
<br> Run Command : python3 lang-model-bigram.py
