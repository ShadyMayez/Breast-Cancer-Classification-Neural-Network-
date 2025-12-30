# Breast Cancer Classification with Neural Networks

## Project Overview and Purpose
This project implements a binary classification model to detect breast cancer (Malignant vs. Benign) using the Wisconsin Breast Cancer diagnostic dataset. It utilizes a fully connected Neural Network to identify patterns in medical features like radius, texture, and perimeter to assist in early diagnosis.

## Key Technologies and Libraries
- **Language**: Python
- **Machine Learning**: `scikit-learn`
- **Deep Learning**: `tensorflow` / `keras`
- **Data Handling**: `numpy`, `pandas`

## Methodology and Workflow
1. **Data Acquisition**: Utilized the `sklearn.datasets.load_breast_cancer()` library.
2. **Standardization**: Features were scaled using `StandardScaler` to ensure the neural network converges efficiently.
3. **Model Architecture**:
   - **Input Layer**: Dense layer matching the number of features.
   - **Hidden Layer**: Dense layer with ReLU activation.
   - **Output Layer**: Dense layer with 2 neurons (representing Malignant and Benign) using Softmax activation.
4. **Training**: Compiled with the `adam` optimizer and `sparse_categorical_crossentropy` loss function.



## Results and Insights
- **Accuracy**: The model achieves high accuracy on the test set (typically >92% depending on the split).
- **Predictive System**: The project includes a functional script where you can input raw medical data points to get an immediate "Malignant" or "Benign" prediction.

## How to Run
1. Upload the `Breast_Cancer_Classification_with_NN.ipynb` to Google Colab or a local Jupyter environment.
2. Install the libraries listed in `requirements.txt`.
3. Run all cells to train the model and test the predictive system.
