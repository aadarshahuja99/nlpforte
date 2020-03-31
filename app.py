from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

filename = 'nlpaa2.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)
@app.route('/predict_api/')
def predict_api():
    review=request.args['rev']
    vect=cv.transform(review).toarray()
    my_prediction = clf.predict(vect)
    return jsonify({"prediction":my_prediction})

#api.add_resource(home,'/')
#api.add_resource(predict_api,'/predict_api/<string:review>')

if __name__ == "__main__":
    app.run(debug=True)
