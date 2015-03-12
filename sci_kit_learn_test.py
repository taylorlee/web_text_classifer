from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# this file for testing and debugging different vectorizers,  classifiers
# not part of web app

movieSentim = datasets.load_files('test')
corpus = movieSentim.data
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
X = vectorizer.fit_transform(corpus)
X_train = vectorizer.transform(movieSentim.data)
logReg = LogisticRegression()
logReg.fit(X_train,  movieSentim.target)
classifier = logReg


def runClassifier(inputString='test.txt', outputString=''):
    read = open(inputString,  'r')
    if (outputString != ''):
        write = open(outputString, 'w')
    testCorp = []
    for line in read:
        line = line[:-1]
        testCorp.append(line)

    read.close()

    Xtest = vectorizer.transform(testCorp)
    pred = classifier.predict(Xtest)
    for i in range(0, len(pred)-1):
        if (outputString == ''):
            print movieSentim.target_names[pred[i]] + ": " + testCorp[i]
        else:
            write.write(movieSentim.target_names[pred[i]] + '\n')
