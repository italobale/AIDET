import os
import numpy as np
from flask import Flask, request, render_template
import pandas as pd
import xgboost as xgb


app = Flask(__name__)

OUTPUT_PATH = './output/'

X_columns = [
    "carat",
    "color",
    "clarity",
    "cut",
    "polish",
    "symmetry",
    "fluorescence",
    "depth",
    "table",
    "width",
    "length",
    "height",
    "polish_na",
    "symmetry_na",
    "fluorescence_na",
    "ASSCHER",
    "CUSHION",
    "DROP",
    "EMERALD",
    "HEART",
    "MARQUISE",
    "OVAL",
    "PEAR",
    "PRINCESS",
    "RADIANT",
    "ROUND",
    "AGS",
    "EGL",
    "GIA",
    "HRD",
    "IGI"
]

shapes = X_columns[15:-6]
certificates = X_columns[-5:]

dict_color = {
    "D": 22,     
    "E": 21,     
    "F": 20,     
    "G": 19,     
    "H": 18,     
    "I": 17,     
    "J": 16,     
    "K": 15,     
    "L": 14,     
    "M": 13,     
    "N": 12,     
    "O": 11,     
    "P": 10,     
    "Q": 9,     
    "R": 8,     
    "S": 7,     
    "T": 6,     
    "U": 5,     
    "V": 4,     
    "W": 3,     
    "X": 2,     
    "Y": 1,     
    "Z": 0,
}

dict_cut = {
    "EX": 4,
    "VG": 3,
    "GO": 2,
    "FA": 1,
    "PO": 0,
}

dict_clarity = {
    "FL": 10,
    "IF": 9,
    "VVS1": 8,
    "VVS2": 7,
    "VS1": 6,
    "VS2": 5,
    "SI1": 4,
    "SI2": 3,
    "I1": 2,
    "I2": 1,
    "I3": 0,
}

dict_fluorescence = {
    "NO": 4,
    "FA": 3,
    "ME": 2,
    "ST": 1,
    "VS": 0,
}

dict_polish = {
    "EX": 4,    
    "VG": 3,    
    "GO": 2,    
    "FA": 1,    
    "PO": 0,    
}

dict_symmetry = {
    "EX": 4, 
    "VG": 3, 
    "GO": 2, 
    "FA": 1, 
    "PO": 0, 
}

# Load xgboost model
model = xgb.Booster()
try:
    model.load_model(os.path.join(OUTPUT_PATH, 'unimodel_v0.1.p'))
except Exception:
    print("Model not found!")


@app.route('/')
@app.route('/index')
def index():

    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    json_ = request.form

    # Generate the output
    X = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)

    for s in shapes:
        if json_["shape"] == s:
            X[s] = 1 
    X["carat"] = float(json_["carat"])
    X["cut"] = int(dict_cut[json_["cut"]])
    X["color"] = int(dict_color[json_["color"]])
    X["clarity"] = int(dict_clarity[json_["clarity"]])
    X["width"] = float(json_["width"])
    X["length"] = float(json_["length"])
    X["height"] = float(json_["height"])
    X["depth"] = float(json_["depth"])
    X["table"] = float(json_["table"])
    X["fluorescence"] = int(dict_fluorescence[json_["fluorescence"]])
    X["polish"] = int(dict_polish[json_["polish"]])
    X["symmetry"] = int(dict_symmetry[json_["symmetry"]])
    for c in certificates:
        if json_["certificate"] == c:
            X[c] = 1 

    dtest = xgb.DMatrix(X)
    preds = model.predict(dtest)
    preds = int(10**preds)

    return render_template(
        "result.html", 
        preds=preds,
        carat=json_["carat"],
        shape=json_["shape"],
        cut=json_["cut"],
        color=json_["color"],
        clarity=json_["clarity"],
        width=json_["width"],
        length=json_["length"], 
        height=json_["height"],
        depth=json_["depth"],
        table=json_["table"],
        fluorescence=json_["fluorescence"],
        polish=json_["polish"],
        symmetry=json_["symmetry"],
        certificate=json_["certificate"],
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=12345, debug=True)
