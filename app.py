from flask import Flask, render_template, request, redirect, send_from_directory
from math import ceil
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models and scaler
rf = joblib.load("models/RandomForest_model.pkl")
lgbm = joblib.load("models/LightGBM_model.pkl")
svm = joblib.load("models/SVM_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load column metadata
feature_cols = joblib.load("models/feature_columns.pkl")
numeric_cols = joblib.load("models/numeric_columns.pkl")

# Original dataset columns (used for TXT uploads)
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]

# --------------------- ROUTES -------------------------

@app.route('/')
def index():
    return render_template("index.html", title="Home")

@app.route('/upload')
def upload_page():
    return render_template("upload.html", title="Upload File")

@app.route('/about')
def about():
    return render_template("about.html", title="About")

@app.route('/manual')
def manual():
    # Only show numeric + few categorical columns for manual form
    input_fields = [c for c in feature_cols if c not in [
        'protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp',
        'label', 'target'
    ]]
    return render_template("manual.html", title="Manual Input", feature_cols=input_fields)

# --------------------- FILE UPLOAD PREDICTION -------------------------

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files["file"]
    filename = file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    if filename.endswith(".txt"):
        df = pd.read_csv(path, names=columns)
    else:
        df = pd.read_csv(path)

    if 'label' in df.columns:
        df['label'] = df['label'].str.rstrip('.')
        df['target'] = (df['label'] != 'normal').astype(int)

    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols)

    df = df.reindex(columns=feature_cols, fill_value=0)

    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df_final = df.copy()

    df["RF_Prediction"] = rf.predict(df_final)
    df["LightGBM_Prediction"] = lgbm.predict(df_final)
    df["SVM_Prediction"] = svm.predict(df_final)

    output_file = os.path.join(UPLOAD_FOLDER, "predicted_output.csv")
    df.to_csv(output_file, index=False)

    # 👉 Auto redirect to pagination
    return redirect("/results?page=1")

@app.route('/results')
def results_paginated():
    page = int(request.args.get("page", 1))
    per_page = 50

    # Load predicted CSV
    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, "predicted_output.csv"))

    # Convert prediction columns to integer
    df['RF_Prediction'] = df['RF_Prediction'].astype(int)
    df['LightGBM_Prediction'] = df['LightGBM_Prediction'].astype(int)
    df['SVM_Prediction'] = df['SVM_Prediction'].astype(int)

    # 👉 KEEP ORIGINAL ORDER – DO NOT SORT
    df = df.reset_index(drop=False)   # keeps the original row number

    total = len(df)
    total_pages = ceil(total / per_page)

    start = (page - 1) * per_page
    end = start + per_page

    df_page = df.iloc[start:end]

    return render_template(
        "pagination.html",
        title="Paginated Results",
        df=df_page,
        page=page,
        total_pages=total_pages
    )

# --------------------- MANUAL ENTRY PREDICTION -------------------------

@app.route('/manual_predict', methods=['POST'])
def manual_predict():
    # Collect manual input from form
    form_data = request.form.to_dict()

    # Convert to DataFrame
    df = pd.DataFrame([form_data])

    # Convert numeric columns to float
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    # Add missing columns
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Scale numeric columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    df_final = df.copy()

    # Predictions
    rf_pred = rf.predict(df_final)[0]
    lgb_pred = lgbm.predict(df_final)[0]
    svm_pred = svm.predict(df_final)[0]

    return render_template(
        "manual_result.html",
        title="Manual Prediction",
        rf=rf_pred,
        lgb=lgb_pred,
        svm=svm_pred
    )

# --------------------- DOWNLOAD FILE -------------------------

@app.route('/uploads/<filename>')
def download(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# --------------------- RUN APP -------------------------

if __name__ == "__main__":
    app.run(debug=True)
