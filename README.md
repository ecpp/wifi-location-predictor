# WiFi Location Predictor

This is a Python script for collecting WiFi signal strength data, training a machine learning model to predict the location based on WiFi signal strength, and making location predictions. The script is designed to run on MacOS and requires administrative privileges.

## Prerequisites

Before using this script, ensure you have the following prerequisites:

1. MacOS: This script is designed to work on MacOS.

2. Python: You need to have Python 3 installed on your system.

3. Required Python Libraries: You can install the required Python libraries using the following command:

   ```
   pip install pandas scikit-learn
   ```

## Usage

1. **Collect WiFi Data (only MacOS)**

   Run this option to collect WiFi signal strength data. You will be prompted to enter a location name. The script will use the `airport` command-line tool to scan for nearby WiFi access points and record their MAC addresses, signal strengths, timestamps, and the specified location. The collected data will be saved in a CSV file with the location name as the filename.

2. **Train Model**

   Use this option to train a machine learning model on the collected WiFi data. You need to provide the filename of the CSV file containing the WiFi data collected earlier. The script will preprocess the data, encode categorical features, split the dataset into training and testing sets, and train a Random Forest Classifier. It will also perform hyperparameter tuning using randomized search and grid search. The results will be displayed, and the best model will be selected for predictions.

3. **Predict Location**

   Once the model is trained, you can use this option to make location predictions based on new WiFi data. Provide the filename of the CSV file containing the WiFi data you want to predict locations for. The script will preprocess the data, encode features, and use the trained model to make predictions. It will display accuracy, precision, and F1 score if the "Location" column is present in the CSV file. Additionally, it will identify the most likely location based on the predictions.

4. **Exit**

   Choose this option to exit the script.

## Important Notes

- The script uses the `airport` command-line tool, which is available on MacOS, to collect WiFi data. Make sure you run the script with administrative privileges (as root) to access this tool.

- The machine learning model is a Random Forest Classifier, and hyperparameter tuning is performed using randomized search and grid search to improve model performance.

- It is important to ensure that the collected WiFi data and the data used for predictions have the same column names and structure.

- The script will display various metrics for model evaluation, including accuracy, precision, and F1 score if applicable.

- If you encounter any issues or have questions, please refer to the script's output messages for guidance.

## How to Run

To run the script, execute it using Python:

```
python wifi_location_predictor.py
```

Follow the on-screen instructions to choose the desired options and provide the necessary input.

**Note:** Ensure that you have the required privileges and permissions to run the script successfully, especially when collecting WiFi data.

## Disclaimer

This script is provided for educational purposes and may require adjustments or modifications for specific use cases or environments. Use it responsibly and ensure compliance with local regulations and privacy laws when collecting and using WiFi data.

**Author:** Eren Copcu
