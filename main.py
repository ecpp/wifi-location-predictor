import os
import subprocess
import csv
import sys
from datetime import datetime
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
is_trained = False
mac_addr_encoded = {}
signal_encoded = {}


def encode_data(data):
    global mac_addr_encoded
    df = pd.DataFrame(data)
    for index, row in df.iterrows():
        if row["MAC Address"] not in mac_addr_encoded:
            mac_addr_encoded[row["MAC Address"]] = len(mac_addr_encoded)
    for index, row in df.iterrows():
        df.at[index, "MAC Address"] = mac_addr_encoded[row["MAC Address"]]

    for index, row in df.iterrows():
        if row["Signal Strength"] not in signal_encoded:
            signal_encoded[row["Signal Strength"]] = len(signal_encoded)
    for index, row in df.iterrows():
        df.at[index, "Signal Strength"] = signal_encoded[row["Signal Strength"]]

    return df


def check_root_or_macos():
    if os.geteuid() != 0:
        print("Please run as root")
        sys.exit(1)
    if sys.platform != "darwin":
        print("Please run on MacOS")
        sys.exit(1)


def collect_data(location):
    result = subprocess.run(
        ['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport', '-s'],
        stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    access_points = output.split("\n")[1:-1]
    data = []
    for access_point in access_points:
        access_point = access_point.split()
        if len(access_point) > 1:
            mac_address = access_point[1]
            signal_strength = access_point[2]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data.append([mac_address, signal_strength, timestamp, location])

    # Write the data to a CSV file
    with open(location + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["MAC Address", "Signal Strength", "Timestamp", "Location"])
        writer.writerows(data)

    print("Data has been written to " + location + ".csv")
    print('\n')


def train_model(filename):
    df = pd.read_csv(filename)
    global clf
    global is_trained
    le = LabelEncoder()

    df = encode_data(df)
    df["Timestamp"] = le.fit_transform(df["Timestamp"])

    X = df[["MAC Address", "Signal Strength"]]
    y = df["Location"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy_nt = accuracy_score(y_test, y_pred)
    precision_nt = precision_score(y_test, y_pred, average='weighted')
    f1_nt = f1_score(y_test, y_pred, average='weighted')
    if accuracy_nt:
        print("*" * 50)
        print("Non-tuned model")
        is_trained = True
        print("Accuracy: {:.2f}%".format(accuracy_nt * 100))
        print("Precision: {:.2f}%".format(precision_nt * 100))
        print("F1 Score: {:.2f}".format(f1_nt))
        print("*" * 50)
        print('\n')
    else:
        print("*** Training failed!")
    print("Randomsearch in progress...")
    param_dist = {"n_estimators": sp_randint(5, 1000),
                  "max_depth": sp_randint(1, 10),
                  "min_samples_split": sp_randint(2, 20),
                  "min_samples_leaf": sp_randint(1, 20),
                  "max_features": sp_randint(1, X_train.shape[1]),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    clf_hyper_rs = RandomForestClassifier()
    random_search = RandomizedSearchCV(clf_hyper_rs, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1)
    random_search.fit(X_train, y_train)
    clf_hyper_rs = RandomForestClassifier(**random_search.best_params_)
    clf_hyper_rs.fit(X_train, y_train)
    y_pred = clf_hyper_rs.predict(X_test)
    accuracy_rs = accuracy_score(y_test, y_pred)
    precision_rs = precision_score(y_test, y_pred, average='weighted')
    f1_rs = f1_score(y_test, y_pred, average='weighted')
    if accuracy_rs:
        if accuracy_rs > accuracy_nt:
            clf = clf_hyper_rs
            print("Randomsearch model is better than non-tuned model")
        print("*" * 50)
        print("Randomsearch Tuned Model trained successfully")
        print("Accuracy: {:.2f}%".format(accuracy_rs * 100))
        print("Precision: {:.2f}%".format(precision_rs * 100))
        print("F1 Score: {:.2f}".format(f1_rs))
        print("*" * 50)
        print('\n')

    else:
        print("Hyperparameter Tuned Model training failed")
        print('\n')
    print("Gridsearch in progress...")
    param_grid = {"n_estimators": [10, 50, 100, 200],
                  "max_depth": [1, 5, 10, 20],
                  "min_samples_split": [2, 5, 10],
                  "min_samples_leaf": [2, 5, 10],
                  "max_features": [1, 2, 3]}
    clf_gs = RandomForestClassifier()
    grid_search = GridSearchCV(clf_gs, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    clf_gs = RandomForestClassifier(**grid_search.best_params_)
    clf_gs.fit(X_train, y_train)
    y_pred = clf_gs.predict(X_test)
    accuracy_gs = accuracy_score(y_test, y_pred)
    precision_gs = precision_score(y_test, y_pred, average='weighted')
    f1_gs = f1_score(y_test, y_pred, average='weighted')
    if accuracy_gs:
        if (accuracy_gs > accuracy_nt and accuracy_gs > accuracy_rs) or (accuracy_gs > accuracy_nt > accuracy_rs):
            print("Gridsearch model is better other models.")
            clf = clf_gs
        print("*" * 50)
        print("Gridsearch Tuned Model trained successfully")
        print("Best parameters: {}".format(grid_search.best_params_))
        print("Accuracy: {:.2f}%".format(accuracy_gs * 100))
        print("Precision: {:.2f}%".format(precision_gs * 100))
        print("F1 Score: {:.2f}".format(f1_gs))
        print("*" * 50)
        print('\n')
    else:
        print("Hyperparameter Tuned Model training failed")
        print('\n')


def predict_location(filename):
    global clf
    print('\n')
    print("*" * 50)
    print("Predicting location...")
    new_data = pd.read_csv(filename)
    new_data = encode_data(new_data)
    X_new = new_data[["MAC Address", "Signal Strength"]]
    location_predictions = clf.predict(X_new)
    if "Location" in new_data.columns:
        accuracy = accuracy_score(new_data["Location"], location_predictions)
        precision = precision_score(new_data["Location"], location_predictions, average='weighted')
        f1 = f1_score(new_data["Location"], location_predictions, average='weighted')
        if accuracy > 0 and precision > 0 and f1 > 0:
            print("Accuracy: {:.2f}%".format(accuracy * 100))
            print("Precision: {:.2f}%".format(precision * 100))
            print("F1 Score: {:.2f}".format(f1))
        else:
            print("Score calculation failed")
    else:
        print("Cannot calculate accuracy, precision and F1 score, because the location column is missing")

    possibilities = set(location_predictions)

    highest_probability = 0
    most_likely_location = ""
    for possibility in possibilities:
        probability = location_predictions.tolist().count(possibility) / len(location_predictions)
        if probability > highest_probability:
            highest_probability = probability
            most_likely_location = possibility

    print("Most likely location: " + str(most_likely_location))
    print("*" * 50)
    print('\n')


def main():
    global is_trained
    while True:
        print("Welcome to the WiFi Location Predictor")
        print("1. Collect WiFi Data (only MacOS)")
        print("2. Train Model")
        print("3. Predict Location")
        print("4. Exit")
        choice = input("Enter your choice:")
        if choice == "1":
            check_root_or_macos()
            location = input("Enter the location: ")
            collect_data(location)
        elif choice == "2":
            filename = input("Enter the filename: ")
            train_model(filename)
        elif choice == "3":
            if is_trained:
                filename = input("Enter the filename: ")
                predict_location(filename)
            else:
                print("*** Model not trained yet or training failed!")
        elif choice == "4":
            print("Bye...")
            exit(0)
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
