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
 
### 11. [Crowdfunding Success Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Crowdfunding%20Campaign%20Success%20Prediction.ipynb)
- **Overview**: Analyzed crowdfunding campaign success using various machine learning models to predict project outcomes based on features like goal amount, category, and duration.
- **Key Techniques**:
  - EDA to explore relationships between features and campaign success.
  - Data preprocessing: handled missing values and outliers; encoded categorical variables.
  - Built and compared models: Random Forest, Naive Bayes, SVM, XGBoost.
  - Feature selection using Recursive Feature Elimination (RFE).
- **Outcome**: SVM achieved the highest accuracy (99.87%), with XGBoost (99.27%) recommended for better efficiency.

### 12. [Marketing Campaign Performance Insights](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Marketing%20Campaign%20Performance%20Insights.ipynb)
- **Overview**: This project analyzes digital marketing campaigns to optimize ROI by examining key metrics such as conversion rates, acquisition costs, and engagement across various channels, campaign types, and audience segments. The analysis includes customer segmentation, time-based trends, and geographic performance to improve future strategies.
- **Key Techniques**:
  - **Data Preprocessing & EDA**: Clean and analyze data to detect patterns.
  - **Correlation Analysis**: Examine relationships between metrics (e.g., clicks, ROI).
  - **Segmentation**: Identify high-performing customer segments.
  - **Geographic & Time-Based Insights**: Optimize marketing based on location and seasonal trends.
  - **Visualization**: Present insights using charts and graphs.
- **Outcome**:
  - **Optimization of Acquisition Costs**: High acquisition costs reduce ROI, suggesting better cost management.
  - **Effective Channels**: Social media and influencer campaigns show strong engagement and ROI.
  - **Customer Segmentation**: Non-English segments (e.g., Spanish, Mandarin) have higher conversion rates.
  - **Geographic Insights**: Cities like Dallas and Chicago offer better ROI and lower costs.
  - **Seasonal Trends**: February and November show dips in performance, requiring improved CTAs.
  - **Campaign Duration**: 30-day campaigns are popular, but 60-day campaigns have higher ROI.
- **Actionable insights**: To focus on reducing acquisition costs, enhancing localization, and leveraging social media for better ROI.

### 13. [Mobile Price Prediction using Linear Regression](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Mobile%20Price%20Prediction.ipynb)
- **Overview**: Predicted mobile prices based on features like RAM, camera quality, and battery using Linear Regression.
- **Key Techniques**:
  - EDA: Analyzed feature distributions, correlations, and identified important factors influencing price.
  - Data preprocessing: Treated categorical variables and handled outliers.
  - Built a Linear Regression model with selected features.
  - Evaluated model performance using R², MAE, MSE, and RMSE.
- **Outcome**: Model explained 85.8% of price variance (R² = 0.858), with MAE of 189. Increasing features improved R² to 0.93.

### 14. [Campus Placement Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Campus%20Placement%20Prediction.ipynb)
- **Overview**: Developed and evaluated predictive models (Logistic Regression, Decision Tree, K-Nearest Neighbors) to classify students' placement status based on academic and extracurricular factors.
- **Key Techniques**:
  - **Data Preprocessing**: Cleaned data by handling missing values and encoding categorical features using Label Encoding.
  - **Feature Selection**: Used correlation matrix and SelectKBest to identify key predictors.
  - **Model Building**: Built and evaluated Logistic Regression, Decision Tree, and KNN models.
  - **Model Evaluation**: Tested model performance using accuracy scores across different parameters.
- **Outcome**: 
  - Logistic Regression performed best with an accuracy of 79.8%.
  - KNN with `k=15` achieved 79.1%, and Decision Tree with `max_depth=3` achieved 77.6%.
  - Logistic Regression was the most consistent and best-performing model for this dataset.

### 15. [Diabetes Diagnosis Based on Patient Health Metrics](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Diabetes%20Diagnosis%20Based%20on%20Patient%20Health%20Metrics.ipynb)
- **Overview**: Predicted diabetes diagnosis (Diabetic, Non-Diabetic, Pre-diabetic) based on patient health metrics like age, BMI, HbA1c, and cholesterol levels using machine learning models.
- **Key Techniques**:
  - Data cleaning: handled missing values, outliers, and standardized columns.
  - Exploratory Data Analysis (EDA) to visualize feature distributions and correlations.
  - Built and evaluated models: Logistic Regression, Random Forest, KNN, SVM.
  - Addressed class imbalance using SMOTE.
- **Outcome**: Random Forest achieved the highest accuracy (98.15%) and F1-score (98.16%), recommended for deployment.

### 16. [Bank Loan Performance Analysis - PowerBI Dashboard](https://app.powerbi.com/view?r=eyJrIjoiODgwOTQ2ZWQtMDEyMy00MjAyLThjMDQtOTE1ZGRhZDcyMGUzIiwidCI6Ijk0NWFlNmVkLWJiZmYtNGM0My05YjRhLWZkNDJmMDRiY2FkZSJ9&authuser=0)
- **Overview**: This project analyzes a loan dataset to understand how borrower details (employment length, income, debt-to-income ratio) and loan characteristics (amount, term, interest rate) affect loan performance. The goal is to provide insights to optimize lending strategies, mitigate credit risk, and enhance portfolio management.
- **Key Techniques**:
  - **Data Transformation**: Cleaned and formatted data, handled missing values, and created new columns like `total_amount_paid` and `delinquency_status`.
  - **Data Modeling**: Established relationships between loan and borrower data.
  - **DAX Measures**: Created calculated columns (e.g., `remaining_installments`) and measures (e.g., `Fully Paid Loan Percentage`).
  - **Visualization**: Built reports on loan performance and borrower profiles using dynamic visuals and key metrics.
- **Outcome**:
  - **Loan Insights**: Analyzed trends in loan performance by amount, status, interest rates, etc.
  - **Borrower Behavior**: Gained insights into borrower demographics, income, and delinquency patterns.
  - **Actionable Insights**: Provided recommendations for improving lending strategies and reducing credit risk.

### 17. [Ecomm Sales Optimization & Customer Satisfaction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Ecomm%20Sales%20Optimization%20%26%20Customer%20Satisfaction.ipynb)  
- **Overview**: Analyzed e-commerce sales data to optimize strategies and improve customer satisfaction by exploring datasets (browsing, sales, customer, feedback) to uncover insights into customer behavior and sales trends.  
- **Key Techniques**:  
  - **Data Merging & Cleaning**: Merged datasets (inner, left join), handled missing values (interpolation, mean/mode imputation), removed duplicates, standardized columns.  
  - **Exploratory Data Analysis**:  
    - Univariate: Distribution analysis, outlier detection.  
    - Bivariate: Correlation heatmaps, scatterplots (sales vs. discounts/ratings).  
    - Advanced: Trend analysis, cohort analysis, sentiment analysis.  
  - **Predictive Modeling**: Built a **Random Forest Regressor** to predict `Sales_Amount`, evaluated with RMSE and R².  
  - **Customer Segmentation**: Applied **K-Means Clustering** to segment customers by purchase behavior and feedback.  
  - **Sales & Feedback Dashboards**: Developed dashboards to visualize sales trends, customer segmentation, and product performance.  
- **Outcome**:  
  - Identified top-performing regions/products.  
  - Highlighted key factors influencing customer satisfaction and feedback.  
  - Built a predictive model for accurate sales forecasting.  
  - Created customer segments for targeted marketing strategies.

### 18. [Loan Approval Prediction and Analysis](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Loan%20Approval%20Prediction%20and%20Analysis.ipynb)
- **Overview**: This project predicts loan approval outcomes using data analysis and machine learning. It explores features like applicant income, credit history, and loan amount, addressing missing values and imbalances, and applying various models to classify loan applications. The goal is to build an accurate model with a focus on the F1 score to handle class imbalance.  
- **Key Techniques**:  
  - **Data Exploration**:  
    - Performed univariate, bivariate, and multivariate analysis on categorical and numerical features.  
    - Visualized distributions of loan amounts, incomes, and credit history.  
    - Analyzed relationships between features like `Property_Area`, `Education`, and `Self_Employed`.  
  - **Data Preprocessing**:  
    - Handled missing values using imputation (KNN, mode, and median).  
    - Removed outliers from `LoanAmount` and applied square root transformation for skewness.  
    - Encoded categorical variables and scaled numerical features.  
  - **Feature Selection**:  
    - Selected key features like `ApplicantIncome`, `LoanAmount`, and `Credit_History` influencing loan approval.  
  - **Modeling**:  
    - Applied classifiers: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, KNN.  
    - Optimized model using **F1 score**, focusing on handling class imbalance.  
    - **Logistic Regression** achieved the best performance, balancing precision and recall.  
- **Outcome**:  
  - Built a **Logistic Regression** model for loan approval prediction with a strong F1 score.  
  - Identified key factors affecting loan approval, such as credit history and income.  
  - Improved model performance through missing value handling and feature scaling.

### 19. [Data-Driven Insights into Realtime Real Estate Price Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Data%20Driven%20Insights%20into%20Realtime%20Real%20Estate%20Price%20Prediction.ipynb)
- **Overview**: This project develops a machine learning model to predict real estate prices based on property features like title, location, carpet area, price, and amenities. It includes data preprocessing, feature engineering, and exploratory data analysis (EDA) to uncover trends and patterns. The goal is to build and evaluate multiple regression models to understand the factors influencing property values.  
- **Key Techniques**:  
  - **Data Loading and Preprocessing**: Loaded the dataset, renamed columns, handled missing values, and removed duplicates.  
  - **Data Cleaning**: Addressed missing values through imputation and removal, standardized area units to square feet, and removed unrealistic outliers.  
  - **Feature Engineering**: Transformed non-numeric columns, extracted useful features, and applied scaling and transformation techniques.  
  - **Exploratory Data Analysis (EDA)**: Visualized distribution patterns, identified relationships between features, and created word clouds for location frequencies.  
  - **Outlier Removal and Skewness Handling**: Applied IQR-based outlier removal and square-root transformations to reduce skewness in key numerical features.  
  - **Model Building**: Evaluated multiple regression models to predict house prices and identify the most important features influencing property values.  
- **Outcome**:  
  - Cleaned dataset with standardized area and price values, improving prediction accuracy.  
  - Visualizations revealed trends such as the most expensive locations and the impact of amenities.  
  - Regression models provided insights and predictions reflecting real-world real estate price dynamics.
