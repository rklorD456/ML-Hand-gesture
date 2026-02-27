
# 🖐️ ML Hand Gesture Recognition (MediaPipe & MLflow)

A complete machine learning pipeline for hand gesture recognition using MediaPipe hand landmarks, scikit-learn, XGBoost, and MLflow for experiment tracking. The project covers data preprocessing, EDA, model training, evaluation, and experiment management.

---

## Project Motivation

Hand gesture recognition enables touchless interfaces, sign language translation, robotics, and AR/VR. This project aims to:
- Build robust ML models for gesture classification
- Explore preprocessing and normalization effects
- Compare and tune multiple algorithms
- Track experiments and results with MLflow

---

## Dataset

- **Source**: MediaPipe hand landmark detection (21 points per hand, 3D)
- **Format**: CSV, 63 features (x, y, z for each landmark) + `label`
- **Classes**: 18 gesture types (e.g., call, dislike, fist, four, like, mute, ok, one, palm, peace, etc.)
- **Samples**: ~25,000
- **Quality**: No missing values, duplicates removed, balanced as possible

---

## Project Structure

- `notebook.ipynb` — Main workflow: EDA, preprocessing, training, evaluation, MLflow logging
- `helper.py` — Helper functions for visualization, preprocessing, and MLflow model logging
- `dataset/hand_landmarks_data.csv` — Preprocessed dataset
- `requirements.txt` — Python dependencies
- `mlruns/` — MLflow experiment logs

---

## Setup & Installation

1. Clone the repository and navigate to the project folder.
2. Install dependencies:
	 ```bash
	 pip install -r requirements.txt
	 ```
3. (Optional) Create and activate a virtual environment.

---

## Usage

### 1. Data Exploration & Model Training
- Open and run `notebook.ipynb` for EDA, preprocessing, model training, and evaluation.
- All experiments and metrics are logged to MLflow automatically.

### 2. MLflow Tracking UI
- Start the MLflow UI to view experiment results:
	```bash
	mlflow ui
	```
- Open [http://localhost:5000](http://localhost:5000) in your browser to explore runs, metrics, and artifacts.

---

## Workflow Summary

### Data Preprocessing
- Check for nulls and duplicates
- Visualize class distribution and sample hand landmarks
- Normalize landmarks (center on wrist, scale by middle finger tip)
- Encode labels
- Stratified train/validation/test split

### Model Training & Evaluation
- Models: Random Forest, SVM, KNN, XGBoost
- Training and validation with accuracy, precision, recall, F1-score
- Hyperparameter tuning (GridSearchCV for XGBoost)
- All results and models logged to MLflow

### Helper Functions
- `plot_single_hand`: Visualize hand landmarks
- `preprocess_landmarks`: Normalize and flatten landmark data
- `train_and_log_model`: Train a model and log to MLflow

---

## Example: Training & Logging a Model

```python
from helper import train_and_log_model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, max_features="sqrt", max_leaf_nodes=6, random_state=42)
train_and_log_model(rf_model, "Random_Forest", X_train, y_train, X_validation, y_validation, rf_params)
```

---

## MLflow Experiment Tracking

- All model parameters, metrics, and artifacts are logged to the `mlruns/` folder
- Use the MLflow UI to compare runs, view metrics, and download models
- Hyperparameter search results (e.g., XGBoost grid search) are tracked as nested runs

---

## Requirements

- Python 
- MediaPipe 
- scikit-learn, XGBoost
- MLflow
- OpenCV, matplotlib, seaborn, pandas, numpy
- See `requirements.txt` for full list

---


---

## MLflow Experiment Tracking & Results

Below are screenshots from the MLflow UI and experiment results for this project:


### 1. Experiments Overview
![MLflow Experiments Overview](assets/experiment%20done.png)
This is MLflow dashboard showing your active experiments:
- **Hand_Gesture_Classification**: Baseline testing and main model comparisons.
- **XGBoost_FineTuning_Detailed**: Focused experiment for XGBoost hyperparameter optimization.
- **Default**: Standard MLflow catch-all experiment.

### 2. Model Comparisons
![MLflow Models Used](assets/mlfow%20models%20used.png)
This view lists the specific runs within the Hand_Gesture_Classification experiment. It shows you tested a variety of architectures:
- XGBoost, Voting Classifier, KNN, SVM, and Random Forest.
All runs originated from the same source (`notebook.ipynb`), allowing you to see which algorithm performed best under the same conditions.

### 3. Hyperparameter Tuning Visualization
![Model Comparison Table](assets/model%20comparison%20table.png)
This Parallel Coordinates Plot visualizes the search for the best hyperparameters within your XGBoost fine-tuning experiment:
- **Parameters**: learning_rate, max_depth, n_estimators
- **Metric**: accuracy_cv
Dark red lines indicate parameter combinations that yielded the highest cross-validation accuracy (around 0.98057).

### 4. Registered Model
![Model Registered](assets/model%20registerateed.png)
This shows the final stage of the workflow. You have "promoted" the best-performing model from your grid search:
- **Model Name**: best_model
- **Registered Version**: xgboost-v1 (v1)
- **Source**: Parent_GridSearch_Run (origin of this version, ready for deployment or production use)

These images demonstrate the experiment tracking, model comparison, hyperparameter tuning, and registration features provided by MLflow in this project.

---

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [MLflow](https://mlflow.org/)


- Hand landmark visualization (plotting joints and connections)
- Custom normalization for MediaPipe landmarks (centering, scaling, flattening)
- Support for multiple ML algorithms: Random Forest, SVM, KNN, XGBoost, LightGBM
- Model tracking and hyperparameter tuning with MLflow
- Jupyter notebook for interactive EDA, training, and evaluation
- Real-time webcam-based gesture prediction


## Project Structure


- `notebook.ipynb` - Main analysis, EDA, preprocessing, model training, evaluation, and experiment tracking
- `helper.py` - Utility functions for visualization, preprocessing, and real-time prediction
- `main.py` - Script for running real-time gesture recognition using webcam
- `dataset/hand_landmarks_data.csv` - Pre-processed hand landmark dataset
- `models/hand_landmarker.task` - MediaPipe model asset for landmark detection
- `requirements.txt` - Python dependencies


## Requirements


- Python 3.8+
- MediaPipe 0.10.21
- scikit-learn, XGBoost
- MLflow for experiment tracking and hyperparameter tuning
- OpenCV, matplotlib, seaborn, pandas, numpy
- See `requirements.txt` for full list


## Installation


```bash
pip install -r requirements.txt
```


## Usage


### 1. Data Exploration & Model Training
- Open and run `notebook.ipynb` for EDA, preprocessing, training, and evaluation.
- Track experiments and tune hyperparameters with MLflow.

### 2. Real-Time Gesture Recognition
- Run `main.py` to start webcam-based gesture prediction.
- Uses the best trained model (`final_model.pkl`) and MediaPipe asset.

---

## EDA & Preprocessing

- **Null Values**: Checked and confirmed no missing values.
- **Class Distribution**: Visualized using count plots; some classes are more frequent than others.
- **Statistical Summary**: Used `describe()` to understand feature ranges.
- **Sample Visualization**: Random samples plotted using `plot_single_hand()` to show landmark structure.
- **Class Samples**: Plotted one sample from each class for visual inspection.
- **PCA Projection**: Applied PCA to visualize gesture clusters; found significant overlap between classes, indicating the challenge of classification.
- **Duplicate Removal**: Dropped duplicate rows to ensure data quality.
- **Feature/Label Split**: Features (landmarks) and labels separated.
- **Custom Normalization**: Used `preprocess_landmarks()` to center landmarks around the wrist, scale by middle finger tip distance, and flatten to 1D array.
- **Label Encoding**: Used `LabelEncoder` to convert gesture names to numeric labels.
- **Train/Validation/Test Split**: Stratified split for robust evaluation.

---

## Helper Functions (Code & Explanation)

### `plot_single_hand(landmarks_flat, label_name="Unknown")`
Plots a hand using landmark coordinates. Shows joints and connections for fingers and palm. Used for visual EDA and debugging.

### `preprocess_landmarks(row)`
Normalizes landmarks: centers around wrist, scales by middle finger tip distance, flattens to 1D array. Ensures model input is consistent and robust to hand position/scale.

### `RealTimeGesturePredictor`
Class for real-time gesture prediction using webcam. Uses MediaPipe Tasks API for landmark detection. Smooths predictions with a buffer to avoid flickering. Draws landmarks and overlays predicted gesture on video stream.

---

## Model Training & Evaluation

### Models Used
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

### Training Workflow
- Models trained on preprocessed data.
- Evaluated using accuracy, precision, recall, and F1-score.
- Training time recorded for each model.
- MLflow used for experiment tracking and hyperparameter tuning.

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions.
- **Precision/Recall/F1**: Weighted averages across classes.
- **Visualization**: Grouped bar chart comparing all models.

---

## Model Results & Comparison


=== Final Model Evaluation Metrics ===



---

## MLflow Experiment Tracking

- MLflow is used to log model parameters, metrics, and artifacts.
- Enables easy comparison and reproducibility.
- Hyperparameter tuning (e.g., grid search for XGBoost) is tracked.

---


## References

- [MediaPipe Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [MLflow Documentation](https://mlflow.org/)

---