# CST8504 Assignment 1 — Prediction Dashboard

## Course Information

**Course:** 26F_CST8504_300 Applying AI Techniques  
**Assignment:** Assignment 1 — Prediction Dashboard

## Team Members

| Student name                 | Student number | Student email              |
| ---------------------------- | -------------- | -------------------------- |
| Olga Durham                  | 040687883      | shap0011@algonquinlive.com |
| Khalid Almahrog              | 041114497      | alma0243@algonquinlive.com |
| Kagma Merleu, Nadia Merville | 041272568      | kamg0008@algonquinlive.com |

## Project Overview

This project is being developed for Assignment 1 of CST8504 — Applying AI Techniques.

The goal of the assignment is to build a small end-to-end machine learning product using a real-world tabular dataset. The project will include data preprocessing, exploratory data analysis, supervised machine learning, an interactive prediction dashboard, version control with Git/GitHub, and containerization with Docker.

For this project, we are using motor vehicle collision data from the City of Toronto Open Data Portal.

## Dataset

**Dataset:** Motor Vehicle Collisions Involving Killed or Seriously Injured Persons  
**Source:** City of Toronto Open Data Portal

Dataset link:

https://open.toronto.ca/dataset/motor-vehicle-collisions-involving-killed-or-seriously-injured-persons/

The dataset contains information about motor vehicle collisions involving people who were killed or seriously injured. It includes collision, road, environmental, location, vehicle, and road-user information.

The dataset has been approved for use in this assignment.

## Proposed Machine Learning Problem

The project uses a **binary classification problem**.

The goal is to investigate whether characteristics of a motor vehicle collision can be used to predict whether the collision results in:

- Fatal Injury
- Non-Fatal Injury

The collision-level field `acclass` is used as the target variable.

The dataset is processed at the collision level so that each `collision_id` represents one modelling record.

Special attention is given to:

- Class imbalance
- Missing values
- Duplicate collision records
- Feature selection
- Potential target leakage
- Proper training, validation, and test splitting

## Development Environment

The project is being developed in **Visual Studio Code** using an isolated **Anaconda environment**.

### Python Environment

The environment was created using Python 3.10:

```bash
conda create -n ai_env python=3.10
conda activate ai_env
```

The configured Python version is:

```
Python 3.10.22
```

VS Code was configured to use the ai_env Conda environment as the Python interpreter.
Main Python Libraries
The main project libraries currently installed include:

- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow
- Streamlit
- Plotly
- joblib
  The installed dependencies and their versions are stored in:

```
requirements.txt
```

The environment dependencies were verified using:

```
python -m pip check
```

and the required Python libraries were successfully tested for import.

## Project Structure

```
assignment1_prediction_dashboard/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│       └── processed_collisions.csv
│
├── models/
│   ├── best_random_forest.pkl
│   ├── preprocessor.pkl
│   └── test_roc_curve.csv
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_processor.py
│   ├── model_trainer.py
│   └── predictor.py
│
├── tests/
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

### Directory Purpose

- app/ — Streamlit dashboard application.
- data/raw/ — Original raw dataset. Raw CSV files are excluded from Git.
- data/processed/ — Cleaned and model-ready datasets.
- models/ — Saved trained models, scalers, encoders, and related artifacts.
- notebooks/ — Exploratory analysis and experimentation.
- src/ — Reusable Python modules for preprocessing, model training, and prediction.
- tests/ — Project tests.
- requirements.txt — Python dependencies required to reproduce the environment.
- Dockerfile — Instructions for building the project Docker image.
- .dockerignore — Files excluded from the Docker build context.
- .gitignore — Files excluded from Git version control.

### GitHub Repository

Project repository:

https://github.com/shap0011/assignment1_prediction_dashboard

## Exploratory Data Analysis

The EDA includes analysis of target class distribution, collision impact type, number of people involved, yearly fatality trends, lighting conditions, road surface conditions, and correlations among selected numeric and binary features.

## Machine Learning

The project uses supervised machine learning to predict whether a collision results in a Fatal or Non-Fatal injury.

Two classification models were compared:

- Decision Tree Classifier
- Random Forest Classifier

The dataset was split into training, validation, and test sets using a stratified 70/15/15 split.

Model preprocessing includes:

- Standardization of numeric features using `StandardScaler`
- One-hot encoding of categorical features using `OneHotEncoder`
- Boolean feature passthrough
- Date-based feature engineering from the collision date and time

Class imbalance was handled using balanced class weights.

### Model Comparison

The Decision Tree was used as the baseline model. A Random Forest model was then trained and tuned using the validation set.

The selected model is a tuned Random Forest Classifier with:

- 200 estimators
- Maximum depth of 10
- Minimum samples split of 5
- Balanced class weights

### Final Test Results

The selected Random Forest achieved the following results on the final test set:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.772 |
| Precision | 0.273 |
| Recall    | 0.375 |
| F1-score  | 0.316 |
| ROC AUC   | 0.687 |

The model performs substantially better than random guessing, but predicting the minority Fatal class remains challenging because the dataset is imbalanced.

### Final Model Comparison

Both models were evaluated on the same test set.

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------: | --------: | -----: | -------: |
| Decision Tree       |    0.625 |     0.187 |  0.500 |    0.272 |
| Tuned Random Forest |    0.772 |     0.273 |  0.375 |    0.316 |

The Tuned Random Forest was selected as the final model because it achieved better accuracy, precision, and F1-score overall. The Decision Tree achieved higher recall for Fatal collisions, showing a trade-off between detecting more Fatal cases and producing more false positives.

The trained model and fitted preprocessing pipeline are saved in:

```text
models/best_random_forest.pkl
models/preprocessor.pkl
```

## Interactive Dashboard

An interactive dashboard has been implemented using Streamlit.

The dashboard includes four main pages:

- **Data Overview**
  - Dataset size
  - Fatal injury rate
  - Adjustable dataset preview
  - Numeric summary statistics

- **Interactive Exploration**
  - Date-range filtering
  - Collision trend by year
  - Fatal injury rate by a user-selected categorical feature
  - Collision outcomes by a user-selected involvement factor

- **Prediction**
  - User input controls for collision characteristics
  - Live Fatal / Non-Fatal injury prediction
  - Non-Fatal and Fatal class probabilities
  - Prediction using the saved preprocessing pipeline and trained Random Forest model

- **Model Performance**
  - Final test metrics
  - Confusion matrix
  - ROC curve
  - Decision Tree vs Tuned Random Forest comparison
  - Final model-selection explanation

The Prediction page uses the saved preprocessing pipeline so new inputs receive the same transformations used during model training.

Run the dashboard locally with:

```bash
streamlit run app/dashboard.py
```

If Streamlit is installed in the active Conda environment, the dashboard can also be started with:

```
python -m streamlit run app/dashboard.py
```

## Docker Deployment

The completed Streamlit dashboard can be run inside a Docker container.

### Build the Docker image

From the project root, run:

```bash
docker build -t assignment1-prediction-dashboard .
```

### Run the container

```
docker run --rm -p 8501:8501 --name assignment1-dashboard assignment1-prediction-dashboard
```

Then open:

```
http://localhost:8501
```

The Dockerized application includes:

- Data Overview
- Interactive Exploration
- Live Prediction
- Model Performance

The trained Random Forest model and preprocessing pipeline are loaded from the models/ directory inside the container, so the dashboard does not retrain the model at startup.

### Docker Image Size

Final image:

```
assignment1-prediction-dashboard:latest
```

- Disk usage: 1.11 GB
- Content size: 252 MB
  The image size was reduced by:
- using python:3.10-slim
- removing unused TensorFlow and Jupyter dependencies
- keeping only runtime dependencies in requirements.txt
- excluding notebooks, raw data, Git files, caches, and local environment files through .dockerignore

### Stop the container

If the container is running in the foreground, press:

```
Ctrl + C
```

Because the container is started with `--rm`, it is automatically removed when it stops.

### Final Verification

The project was cloned into a new empty directory and the Docker image was rebuilt and run successfully from the fresh clone. The full Streamlit dashboard, date filters, live predictions, and model performance page were verified inside the container.

## Project Status

Current project progress:

- [x] GitHub repository created
- [x] Initial project structure created
- [x] Raw dataset selected
- [x] Raw dataset excluded from Git
- [x] Python 3.10 Conda environment created
- [x] VS Code configured to use the project environment
- [x] Required Python libraries installed
- [x] requirements.txt generated
- [x] Dataset approval confirmed with professor
- [x] Data inspection and preprocessing
- [x] Exploratory data analysis
- [x] Streamlit Data Overview page
- [x] Streamlit Interactive Exploration page
- [x] Machine learning model training and comparison
- [x] Model preprocessing pipeline saved
- [x] Final trained Random Forest model saved
- [x] Prediction page model integration
- [x] Model Performance page
- [x] Docker image and container
- [x] Final testing and documentation
