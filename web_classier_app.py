from flask import Flask
app = Flask(__name__)

from flask import render_template
from flask import request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


vectorizer, classifier, movie_sentim = None, None, None


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/train", methods=['POST'])
def train():
    global vectorizer, classifier, movie_sentim
    movie_sentim = datasets.load_files('test')
    corpus = movie_sentim.data
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=1)
    vectorizer.fit_transform(corpus)
    X_train = vectorizer.transform(movie_sentim.data)
    classifier = LogisticRegression()
    classifier.fit(X_train,  movie_sentim.target)
    return "Success"


@app.route("/predict", methods=['POST'])
def predict():
    testCorp = request.form['text'].split('\n')
    Xtest = vectorizer.transform(testCorp)
    pred = classifier.predict(Xtest)
    outText = ""
    for i in range(0, len(pred)):
        outText += movie_sentim.target_names[pred[i]] + \
            ": " + testCorp[i] + "\n"
    return outText


if __name__ == "__main__":
    app.debug = True
    app.run()
