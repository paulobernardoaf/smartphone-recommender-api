import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from flask import request
from flask_cors import CORS
import json

import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

lab_enc = LabelEncoder()

full_dataset = pd.read_csv("Flipkart_output.csv", sep=",")

def splitDataFrameList(df,target_column,separator):
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s.strip()
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df

def buildDataset(full_dataset):

    new_dataset = full_dataset.drop(columns=["ImageUrl", "Price in Rupees", "Camera_details", "Processor"])

    storage = new_dataset["Storage_details"].str.split("|", n=2, expand=True)

    new_dataset["Color"] = new_dataset["Name"].str.extract(pat = '.*\((.*)\).*')
    new_dataset["Color"] = new_dataset["Color"].str.rpartition(", ", expand=True)

    new_dataset = new_dataset.dropna()

    new_dataset = splitDataFrameList(new_dataset, "Color", ",")

    new_dataset["Screen_size"] = new_dataset["Screen_size"].str.extract(pat = '.*\((.*)\).*')
    new_dataset["Screen_size"] = new_dataset["Screen_size"].str.rpartition(" inch", expand=True)
    
    new_dataset["Battery_details"] = new_dataset["Battery_details"].str.rpartition(" m", expand=True)

    new_dataset["Ram_storage"] = storage[0]
    new_dataset["Ram_storage"] = new_dataset["Ram_storage"].str.rpartition(" GB", expand=True)
    
    new_dataset["Mem_storage"] = storage[1]
    new_dataset["Mem_storage"] = new_dataset["Mem_storage"].str.rpartition(" GB", expand=True)
    
    new_dataset = new_dataset.dropna()

    new_dataset["Screen_size"] = new_dataset["Screen_size"].astype("float32")
    new_dataset["Battery_details"] = new_dataset["Battery_details"].astype("int32")
    new_dataset["Ram_storage"] = new_dataset["Ram_storage"].astype("float32")
    new_dataset["Mem_storage"] = new_dataset["Mem_storage"].astype("int32")

    new_dataset.drop(columns = "Storage_details", inplace=True)
    results = new_dataset["Name"].apply(lambda x : x.split(" ")[0])
    new_dataset.drop(columns = "Name", inplace=True)

    colors = new_dataset["Color"].unique()

    new_dataset.drop(columns = "Color", inplace=True)

    results = lab_enc.fit_transform(results)

    return new_dataset, results, colors

def run_classifier(screen, ram, storage, battery):
    new_dataset, y, colors = buildDataset(full_dataset)

    scaler = StandardScaler()
    X = scaler.fit_transform(new_dataset)

    # # Training the dataset
    from sklearn.tree import DecisionTreeClassifier
    lin = DecisionTreeClassifier()

    from sklearn.model_selection import cross_validate
    result = cross_validate(lin, X, y, cv=10, scoring="accuracy", return_estimator=True)

    best_instance = np.argmax(result['test_score'])

    best_classifier = result["estimator"][best_instance]

    prediction = best_classifier.predict(scaler.transform([[screen, battery, ram, storage]]))

    brand = lab_enc.inverse_transform(prediction)[0]

    string = str(int(float(ram))) + " GB RAM | " + str(storage) + " GB ROM"

    phones = full_dataset.loc[full_dataset["Name"].str.contains(brand) & full_dataset["Storage_details"].str.contains(string) & (new_dataset["Screen_size"].apply(np.isclose, b=float(screen), atol=0.1) ) & (new_dataset["Battery_details"].apply(np.isclose, b=int(battery), atol=600))]
    phones = phones.drop(columns=["Price in Rupees", "Processor"])

    return phones, max(result['test_score']), brand

def build_info():
    new_dataset, y, colors = buildDataset(full_dataset)
    screens = ["%.1f" % number for number in new_dataset["Screen_size"].unique()]
    rams = ["%.1f" % number for number in new_dataset["Ram_storage"].unique()]
    mems = new_dataset["Mem_storage"].unique()
    batteries = new_dataset["Battery_details"].unique()

    return list(dict.fromkeys(screens)), list(dict.fromkeys(rams)), mems, batteries


def generate_result(screen, ram, storage, battery):
    yield run_classifier(screen, ram, storage, battery)

def generate_info():
    yield build_info()

@app.route('/phones', methods=['GET'])
def phones():
    screen = request.args.get('screen')
    ram = request.args.get('ram')
    storage = request.args.get('storage')
    battery = request.args.get('battery')
    result, accuracy, brand = next(generate_result(screen, ram, storage, battery))
    return json.dumps({"phones": result.to_dict(orient='records'), "accuracy": accuracy, "brand": brand})

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/info')
def info():
    screens, rams, mems, batteries = next(generate_info())
    screens.sort()
    rams.sort()
    batteries.sort()
    mems.sort()
    return json.dumps({"screens":screens,"rams": rams,"mems": mems.tolist(),"batteries": batteries.tolist()})


if __name__ == '__main__':
    app.run(threaded=True, port=5000, debug=True)


