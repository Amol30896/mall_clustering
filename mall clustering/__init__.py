import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
@app.route('/train_data')
def train():
    df = pd.read_csv("Mall_Customers.csv")
    x = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    x = np.array(x)
    kmean = KMeans(n_clusters=5, init="k-means++", random_state=42)
    kmean.fit(x)
    joblib.dump(kmean, 'train.pkl')
    return "Model train succesfully"

@app.route("/test_data", methods=['POST'])
def test():
    pkl_file=joblib.load("train.pkl")
    test_data=request.get_json()
    f1 = test_data['Annual Income (k$)']
    f2 = test_data['Spending Score (1-100)']
    my_test_data = [f1,f2]
    my_data_array = np.array(my_test_data)
    test_array  =my_data_array.reshape(1,2)
    df_test = pd.DataFrame(test_array,columns=['Annual Income (k$)','Spending Score (1-100)'])
    pred=pkl_file.predict(df_test)
    return "Model test succesfully"
app.run(port=5000)
