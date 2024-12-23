# DS-Portfolio
My portfolio includes Data Science projects demonstrating expertise in Machine Learning, Deep Learning, and data analysis. These projects cover end-to-end workflows, from data preprocessing and feature engineering to model development and evaluation, reflecting a strong foundation in data-driven decision-making and problem-solving.


## Projects

### 1. [Credit Card Fraud Detection System](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Credit%20Card%20Fraud%20Detection.ipynb)
- **Overview**: Developed a real-time fraud detection system using machine learning models like Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **Key Techniques**:
  - Extensive exploratory data analysis (EDA) to analyze transaction patterns and detect anomalies.
  - SMOTE for handling class imbalance.
  - Hyperparameter tuning to optimize model performance.
- **Outcome**: Developed a model with promising potential for fraud detection. While the model effectively reduces false positives, improving recall and overall fraud detection accuracy may be limited by the inherent class imbalance in the dataset.

### 2. [Stock Price Forecasting with LSTM](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Stock%20Price%20Analysis%20and%20Forecasting.ipynb)
- **Overview**: Created a stock price prediction model using Long Short-Term Memory (LSTM) networks with historical market data.
- **Key Techniques**:
  - Time-series analysis to uncover trends and seasonality in stock prices.
  - Financial modeling and correlation analysis between major tech companies.
- **Outcome**: Developed a forecasting model that generates actionable insights for investment strategies and price volatility predictions.

### 3. [Image Captioning with Deep Learning](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Image%20Captioning%20System.ipynb)
- **Overview**: Built an image captioning model using VGG16 for feature extraction and LSTM for natural language generation.
- **Key Techniques**:
  - EDA on the Flickr8k dataset to analyze image-text relationships.
  - Tokenization, sequence padding, and vocabulary management for generating captions.
- **Outcome**: Developed a scalable image captioning system for generating accurate descriptions for unseen images.

### 4. [Movie Recommendation System](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Movie%20Recommendation%20System.ipynb)
- **Overview**: Built a recommendation system using demographic, content-based, and collaborative filtering techniques with TMDB's dataset.
- **Key Techniques**:
  - EDA to identify key features such as genre, cast, and user ratings.
  - Integrated weighted rating formulas and collaborative filtering for personalized recommendations.
- **Outcome**: Delivered a scalable solution for personalized movie recommendations for streaming platforms.

### 5. [Sentiment Analysis of E-Commerce Reviews](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Sentiment%20Analysis%20of%20E-Commerce%20Clothing%20Reviews.ipynb)
- **Overview**: Developed a sentiment analysis model using Naive Bayes algorithms (Bernoulli and Multinomial) to classify customer reviews from an e-commerce platform. Additionally, built models to predict customer feedback using various machine learning techniques, including Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest, Support Vector Classifier (SVC), Naive Bayes, and XGBoost.
- **Key Techniques**:
  - EDA to analyze review patterns, identify sentiment trends, and visualize feature distributions.
  - Performed data preprocessing to handle missing values, outliers, and skewness, while utilizing feature engineering to improve model accuracy.
  - Created word clouds to visualize frequent terms and enhance feature understanding.
- **Outcome**: Provided actionable insights for product recommendations and customer satisfaction improvements.

### 6. [Customer Churn Analysis and Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Customer%20Churn%20Analysis%20and%20Prediction.ipynb)
- **Overview**: Analyzed customer churn in the telecom industry using the Telco customer dataset to predict customer churn. The project built and evaluated several machine learning models to identify the factors influencing churn and improve customer retention strategies.
- **Key Techniques**:
  - EDA to visualize churn patterns and feature distributions.
  - Data preprocessing: handled missing values, encoded categorical variables ; Feature selection: `SelectKBest`.
  - Built and compared models: Logistic Regression, Decision Trees, Random Forest, SVM, XGBoost, Naive Bayes.
  - Hyperparameter tuning using Grid Search for Random Forest.
- **Outcome**: Logistic Regression achieved the highest accuracy (80.01%) and F1 score (0.6053). Key insights highlighted the importance of tenure and contract type in predicting churn.

### 7. [Customer Segmentation using KMeans Clustering](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Customer%20Segmentation%20using%20KMeans%20Clustering.ipynb)
- **Overview**: Applied k-Means clustering to segment customers based on their demographic and spending behaviors. The objective was to uncover patterns and inform targeted marketing strategies to improve customer engagement and satisfaction.
- **Key Techniques**:
  - Data preprocessing: handled missing values and encoded categorical variables.
  - Exploratory Data Analysis (EDA) to identify correlations and visualize data distributions.
  - Performed feature scaling using `StandardScaler` to standardize data before clustering.
  - Determined optimal number of clusters (K=5) using the Elbow method.
  - Applied k-Means clustering and analyzed the resulting clusters for patterns.
  - Statistical tests: ANOVA and Chi-squared tests to analyze relationships between clusters and customer attributes.
- **Outcome**: Identified five distinct customer segments, ranging from high spenders to low-income savers. The analysis provides valuable insights for developing personalized marketing strategies.

### 8. [Clustering Countries by Socio-Economic and Health Indicators using Hierarchical Clustering](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Clustering%20Countries%20by%20Development%20Indicators%20using%20Hierarchical%20Clustering.ipynb)
- **Overview**: This project applies hierarchical clustering to categorize countries into three clusters (Developed, Developing, and Least Developed) based on socio-economic and health indicators. The objective is to gain insights into the differences between countries at various stages of development and inform targeted policy interventions.
- **Key Techniques**:
  - Data Preprocessing: Handled missing values, performed feature selection, and created a correlation matrix to understand relationships between socio-economic and health indicators.
  - Exploratory Data Analysis (EDA): Visualized correlations among key features such as GDP per capita, income, child mortality, life expectancy, and fertility rates.
  - Performed feature scaling using `StandardScaler` to standardize the data for clustering.
  - Used **Dendrogram** to visually determine the optimal number of clusters, which suggested either 2 or 3 clusters.
  - Applied the **Elbow Method** and **Silhouette Score** to determine the optimal number of clusters (3 clusters).
  - Performed **Agglomerative Hierarchical Clustering** and labeled the countries into 3 clusters.
  - Visualized the clusters using scatter plots to analyze relationships between features like income vs. GDP per capita, life expectancy vs. child mortality, and child mortality vs. GDP per capita.
- **Outcome**: The analysis successfully categorized countries into three clusters:
  - **Developed Countries**: High income, high GDP per capita, low child mortality, high life expectancy.
  - **Developing Countries**: Moderate income, moderate GDP per capita, moderate health outcomes.
  - **Least Developed Countries**: Low income, low GDP per capita, high child mortality, low life expectancy.
  - The findings can be used to inform global development strategies and prioritize interventions.

### 9. [Life Expectancy (WHO) Analysis and Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Life%20Expectancy%20(WHO)%20Analysis%20and%20Prediction%20-%20Classification%20and%20Regression%20Models.ipynb)
- **Overview**: This project analyzes and predicts life expectancy across countries using **regression** and **classification models**. It focuses on identifying the factors influencing life expectancy and classifying countries as "developed" or "developing."
  1. **Predict Life Expectancy**: Use regression models to estimate life expectancy based on health, economic, and demographic features.
  2. **Classify Development Status**: Use classification models to categorize countries as "developed" or "developing."
- **Key Techniques**:
  - **Data Preprocessing**: Handled missing values via mean/median imputation and label encoding for categorical variables.
  - **Exploratory Data Analysis (EDA)**: Visualized distributions and relationships of life expectancy, GDP, and other variables.
  - **Feature Selection**: Applied **SelectKBest** with **ANOVA F-test** for classification feature selection.
  - **Models**:
    - **Regression Models**: Linear Regression, Decision Trees, K-Nearest Neighbors, Random Forest, Gradient Boosting, AdaBoost, XGBoost, Support Vector Regression (SVR).
    - **Classification Models**: Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Gaussian Naive Bayes, Support Vector Classifier (SVC).
  - **Model Tuning**: Optimized performance using **GridSearchCV**.
  - **Evaluation**: **R²** for regression and **F1 Score**, **Accuracy**, **Precision**, **Recall**, **ROC-AUC** for classification.
- **Outcome**:
  - **Regression Outcome**: Key factors influencing life expectancy include **Adult Mortality**, **Income Composition of Resources**, and **Schooling**.
  - **Classification Outcome**: **XGBoost** achieved the best performance in distinguishing "Developed" vs. "Developing" countries.
  - **Key Insights**:
    - Lower GDP countries face challenges in improving life expectancy.
    - Developed countries have higher life expectancy, while developing nations show more variability.
  - **Best Model**: **XGBoost** was the best model for both **regression** (predicting life expectancy) and **classification** (development status) after hyperparameter tuning.

### 10. [Handwritten Digit Recognition with CNN](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Handwritten%20Digit%20Recognition%20with%20CNN.ipynb)
- **Overview**: This project uses a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST** dataset, consisting of 60,000 training images and 10,000 test images of 28x28 grayscale digits (0-9).
- **Key Steps**:
  - **Data Preprocessing**: Normalize pixel values to [0, 1] by dividing by 255.
  - **PCA**: Applied PCA to visualize the dataset in 2D, showing clustering of digits based on their features.
  - **Reshaping Data**: Reshaped input images for CNN (28x28x1).
  - **One-Hot Encoding**: Converted labels to one-hot encoding for multi-class classification.
  - **CNN Model**: Built a CNN with two convolutional layers, max-pooling layers, and a dense output layer.
  - **Training**: Trained the model for 10 epochs using Adam optimizer and categorical crossentropy loss.
  - **Evaluation**: Achieved high accuracy on the test set.
- **Outcome**:
  - **Test Accuracy**: The model achieved a high test accuracy, demonstrating the effectiveness of CNNs in digit recognition.
  - **Key Insights**:
    - CNNs are well-suited for image classification tasks, particularly with datasets like MNIST.
    - Proper data preprocessing (normalization, reshaping) and one-hot encoding significantly improve model performance.
  - **Best Model**: CNN with **two convolutional layers**, max-pooling, and a dense output layer achieved optimal performance in classifying MNIST digits.

