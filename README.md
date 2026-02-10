# Hand Gesture Recognition ML Project

A machine learning project for recognizing hand gestures using MediaPipe hand landmarks and various classification algorithms.

## Overview

This project trains ML models to classify hand gestures based on 21 hand landmark coordinates (x, y, z) extracted using MediaPipe. The dataset contains pre-processed hand landmark data that can be used to train and evaluate different machine learning models.

## Features

- Hand landmark visualization
- Custom normalization for MediaPipe landmarks
- Support for multiple ML algorithms (scikit-learn, XGBoost, LightGBM)
- Model tracking with MLflow
- Jupyter notebook for interactive analysis

## Project Structure

- `notebook.ipynb` - Main analysis and model training workflow
- `helper.py` - Utility functions for visualization and preprocessing
- `dataset/` - Hand landmarks dataset (CSV format)
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.8+
- MediaPipe 0.10.21
- scikit-learn, XGBoost, LightGBM
- MLflow for experiment tracking
- See `requirements.txt` for full list

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Open and run the `notebook.ipynb` to explore the data and train models.

## Dataset

The dataset consists of hand landmarks with 63 features (21 landmarks × 3 coordinates) extracted from hand gesture images.
