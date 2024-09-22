# Phishing Detection using URL Embeddings from DistilBert and other URL features.
## Project Overview

This project focuses on developing a phishing URL detection system using a combination of machine learning models and URL embeddings. With phishing attacks becoming increasingly sophisticated, the project aims to create a robust classification system capable of identifying phishing websites in real-time. By leveraging advanced techniques such as DistilBERT for URL embeddings and popular classifiers like XGBoost, Random Forest, and Artificial Neural Networks (ANN), this system seeks to improve detection accuracy and generalization across various datasets.

## Features

- **Phishing URL Detection**: Classifies URLs as phishing or legitimate.
- **Multiple Classifiers**: Comparative analysis using XGBoost, ANN, Random Forest, Logistic Regression, and SVM.
- **Feature Engineering**: Advanced techniques to extract and combine features from URLs, including embedding models.
- **DistilBERT for URL Embedding**: Utilizes a distilled version of BERT to encode URLs into feature-rich embeddings for improved detection accuracy.
- **Hyperparameter Tuning**: Optimized models with tuned hyperparameters to enhance model performance and prevent overfitting.

## Data

The project uses publicly available phishing datasets from [Mendeley Data](https://data.mendeley.com/datasets/c2gw7fy2j4/3) and other sources. The datasets are preprocessed to extract relevant features such as URL structure, domain details, and lexical attributes. URL embeddings are generated using the DistilBERT model to capture the semantic meaning of the URLs.

## Models Implemented

1. **Artificial Neural Network (ANN)**:
   - Architecture: Multiple hidden layers with ReLU activation functions.
   - Loss Function: Binary cross-entropy.
   - Optimizer: Adam with learning rate of 0.001.
   - Early stopping used to avoid overfitting.

2. **XGBoost**:
   - Objective: Binary classification with logloss evaluation.
   - Tuned hyperparameters: Learning rate, max depth, subsample, colsample_bytree, and regularization (L1, L2).

3. **Random Forest**:
   - A robust ensemble model.
   - Feature importance-based selection for better generalization.

4. **SVM (Support Vector Machine)**:
   - Default classifier for baseline comparison.

5. **Logistic Regression**:
   - Provides a simple linear classification baseline.

## How It Works

1. **Data Preprocessing**:
   - Tokenizes URLs and extracts key lexical and structural features.
   - URL embeddings are generated using DistilBERT, converting the URLs into numerical vectors for the models.

2. **Model Training**:
   - Models are trained using the extracted features and embeddings.
   - Cross-validation and early stopping are applied to avoid overfitting.
   
3. **Model Evaluation**:
   - Metrics used: Accuracy, precision, recall, F1-score.
   - Hyperparameter tuning is performed to improve model performance.



## Results

The XGBoost model achieved the highest accuracy, outperforming other classifiers in terms of generalization and precision. The use of DistilBERT embeddings significantly improved detection accuracy by capturing the semantics of URLs.

## Future Work

- **Improve Feature Engineering**: Explore additional URL features to enhance model performance.
- **Deploy a Real-Time System**: Develop a live phishing detection API.
- **Explore More NLP Models**: Test other transformer-based models like BERT or GPT for URL embeddings.

## Acknowledgements

Special thanks to the mentors and professors for guidance throughout this project. The datasets used in this project are provided by the Mendeley Data platform.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
