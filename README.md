# Enhancing Credit Score Prediction

This project focuses on **Credit Score Prediction** using **Random Forest Classifier** to assist financial institutions in assessing creditworthiness efficiently and accurately. By leveraging data visualization and machine learning techniques, the model classifies individuals' credit scores into three categories: **Poor**, **Standard**, and **Good**, offering valuable insights into the key factors influencing credit ratings.

---

## Table of Contents

1. [Abstract](#abstract)  
2. [Features](#features)  
3. [Technologies Used](#technologies-used)  
4. [Dataset Description](#dataset-description)  
5. [Project Workflow](#project-workflow)  
6. [Results](#results)  
7. [Future Scope](#future-scope)  
8. [Installation](#installation)  
9. [Usage](#usage)  
10. [License](#license)  

---

## Abstract

Credit scoring is essential for financial risk management. This project employs a **Random Forest Classifier** to predict credit scores using financial and behavioral attributes like income, debt, and payment history. The results demonstrate an **accuracy of 84.14%**, and the project emphasizes key factors such as income levels, credit utilization, and payment habits.

---

## Features

- Predict credit scores into three categories: Poor, Standard, and Good.
- Data preprocessing, including handling missing values and encoding categorical variables.
- Data visualization using **box plots** for feature impact analysis.
- Performance evaluation using:
  - **Accuracy Score**
  - **Classification Report (Precision, Recall, F1-Score)**
  - **Confusion Matrix**

---

## Technologies Used

- **Python**: Core programming language.  
- **Libraries**:  
  - `scikit-learn` for machine learning.  
  - `pandas` and `numpy` for data manipulation.  
  - `matplotlib` and `seaborn` for visualization.  

---

## Dataset Description

The dataset includes the following key features:  
- **Financial Attributes**: Annual income, monthly salary, credit utilization, number of loans, etc.  
- **Behavioral Attributes**: Number of delayed payments, delay from due date, credit history age, etc.  
- **Target Variable**:  
  - 0: Poor  
  - 1: Standard  
  - 2: Good  

---

## Project Workflow

1. **Data Preprocessing**:  
   - Handling missing values.  
   - Encoding categorical variables.  
   - Feature scaling as needed.  

2. **Model Training**:  
   - Random Forest Classifier.  
   - Train-test split for evaluation.  

3. **Visualization**:  
   - Box plots for exploring relationships between features and credit scores.  

4. **Performance Metrics**:  
   - Accuracy, Precision, Recall, F1-Score.  
   - Confusion Matrix for error analysis.  

---

## Results

- **Overall Accuracy**: 84.14%  
- **Class-wise Performance**:  
  - Poor: Precision: 0.81, Recall: 0.82, F1-Score: 0.81  
  - Standard: Precision: 0.82, Recall: 0.87, F1-Score: 0.85  
  - Good: Precision: 0.87, Recall: 0.83, F1-Score: 0.85  

- **Key Insights**:  
  - Higher income and lower credit utilization generally correlate with better credit scores.  
  - Delayed payments significantly impact creditworthiness.  

---

## Future Scope

- Incorporate additional features for improved predictions.  
- Explore hybrid machine learning models.  
- Implement hyperparameter tuning for model optimization.  
- Extend the project to support real-time credit score prediction.  

---

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/HIMANSUAGRAWAL/Credit-Score-Prediction.git
   cd Credit-Score-Prediction
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:  
   ```bash
   jupyter notebook Credit_Score_Prediction.ipynb
   ```

---

## Usage

1. Load the dataset and preprocess it.  
2. Train the Random Forest Classifier using the provided workflow.  
3. Use the trained model to classify new data points.  
4. Visualize key features influencing credit scores using the provided box plot scripts.  

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Author  

**Himansu Agrawal**  
- 📧 Email: [himansubansal1701@gmail.com](mailto:himansubansal1701@gmail.com)  
- 🌐 GitHub: [HIMANSUAGRAWAL](https://github.com/HIMANSUAGRAWAL)  
- 🔗 LinkedIn: [Himansu Agrawal](https://www.linkedin.com/in/himansu-agrawal-45410333b/)  
- X: [@HimansuBan73216](https://x.com/HimansuBan73216)  
- Instagram: [@_himansubansal_](https://www.instagram.com/_himansubansal_/)

Feel free to fork this repository, open issues, or contribute enhancements! 😊
