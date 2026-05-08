# DS-Portfolio

My portfolio includes Data Science and AI projects demonstrating expertise in Generative AI, Agentic AI, Machine Learning, Deep Learning, and data analysis. These projects cover end-to-end workflows — from data preprocessing and feature engineering to LLM-powered pipelines and autonomous agent orchestration — reflecting a strong foundation in data-driven decision-making and modern AI engineering.

---

## Projects

### 1. [End-to-End RAG System with Retrieval Optimization & Evaluation](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/End-to-End%20RAG%20System%20with%20Retrieval%20Optimization.ipynb)
- **Overview**: A production-oriented Retrieval-Augmented Generation (RAG) pipeline built using open-source LLMs for financial document intelligence over NVIDIA's 2025 10-K filing. Progresses from a baseline RAG pipeline to a fully optimized system with hybrid retrieval, reranking, and RAGAS-based evaluation.
- **Key Techniques**:
  - Semantic chunking with overlap optimization (chunk size 800, overlap 100) for dense financial documents.
  - FAISS vector store with BAAI/bge-small-en-v1.5 sentence embeddings.
  - Hybrid retrieval combining BM25 keyword search and semantic vector search via EnsembleRetriever.
  - FlashRank cross-encoder reranking for improved context precision.
  - RAGAS evaluation framework measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
  - Hallucination stress testing with adversarial queries (non-existent products, unavailable financial data).
  - 4-bit quantization (bitsandbytes NF4) for efficient local Mistral-7B inference on Colab GPU.
- **Outcome**: Hybrid retrieval with reranking demonstrably improved precision on financial line-item extraction, correctly surfacing specific figures (e.g., $4.5B H20 inventory charge) that standard semantic search deprioritized. RAGAS baseline scores — Faithfulness: 0.82, Answer Relevancy: 0.78, Context Precision: 0.75, Context Recall: 0.80.
- **Tech Stack**: Mistral-7B-Instruct, LangChain, FAISS, HuggingFace Transformers, BAAI/bge-small-en-v1.5, RAGAS, FlashRank, BitsAndBytes

---

### 2. [LLM-Powered Document Assistant](https://github.com/ArchanaAnalytics/DS-Portfolio/tree/main/Projects/LLM-Powered_Document_Assistant)
- **Overview**: A FastAPI-based document Q&A assistant built using RAG principles for grounded, citation-attributed interaction over enterprise documents. Designed for production reliability with async request handling, caching, and rate limiting.
- **Key Techniques**:
  - FastAPI backend with async request handling and retry logic for resilient LLM calls.
  - FAISS-based document retrieval with RecursiveCharacterTextSplitter (chunk size 800, overlap 100).
  - Query normalization for improved cache hit rates, reducing redundant LLM API calls.
  - Per-IP rate limiting (10 requests/minute) with a custom RateLimiter module.
  - Source attribution — every response cites the originating document filename.
  - Prompt-grounded responses: LLM is instructed to answer only from retrieved context.
  - Modular architecture: separate app/, docs/, frontend/ directories for scalability.
- **Outcome**: Production-ready document assistant with measurable latency reduction via caching, graceful error handling under concurrent load, and a clean chat UI served from the same FastAPI app.
- **Tech Stack**: FastAPI, LangChain, OpenAI API (GPT-4o-mini), FAISS, Python AsyncIO, Pydantic

---

### 3. [Autonomous Multi-Agent Research Analyst](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Autonomous%20Multi-Agent%20Research%20Analyst%20using%20CrewAI.ipynb)
- **Overview**: A production-oriented Agentic AI pipeline built using the CrewAI orchestration framework. Four specialized autonomous agents — Researcher, Analyst, Writer, and Editor — collaborate sequentially to perform real-time web research, strategic analysis, report generation, and editorial validation on any given topic.
- **Key Techniques**:
  - Custom DuckDuckGoSearchTool wrapping LangChain's search integration for real-time web data.
  - Four specialized agents each with distinct role, goal, backstory, and tool access.
  - Sequential multi-agent orchestration: each agent's output becomes the next agent's context.
  - Dynamic topic injection via `{topic}` input variables — fully reusable pipeline.
  - SWOT analysis and trend prioritization embedded into the Analyst agent's task specification.
  - Editorial validation pass by a dedicated Reviewer agent before final output.
- **Outcome**: System autonomously produced a structured, publication-ready "2025 Agentic AI Trends" report with real-time sourced research, strategic SWOT analysis, and editorial validation — all without human intervention after kickoff.
- **Tech Stack**: CrewAI, Groq API (LLaMA 3.3 70B), LangChain, DuckDuckGo Search

---

### 4. [Agentic AI Data Analysis Agent](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Agentic%20AI%20Data%20Analysis%20Agent%20using%20LangGraph.ipynb)
- **Overview**: A fully autonomous data analysis agent built using LangGraph's stateful graph orchestration. The agent autonomously profiles a raw dataset, architects an EDA strategy, writes and executes Python code, interprets results, and synthesizes a business-ready analytical report — simulating an end-to-end AI Data Scientist workflow.
- **Key Techniques**:
  - LangGraph StateGraph with a shared `PipelineState` TypedDict flowing through all nodes.
  - Five specialized nodes: Dataset Profiler → Strategic Planner → Code Execution Engine → Insight Interpreter → Technical Reporter.
  - LLM-driven code generation with sandboxed `exec()` and stdout capture for runtime EDA execution.
  - Exception logging within state for debugging and retry-readiness.
  - Results grounded in actual runtime statistics — no hardcoded domain assumptions.
- **Outcome**: Agent autonomously identified key churn drivers from the Telco dataset — Fiber optic customers churn at 41.8% vs 7.4% for no-internet customers; electronic check payment method associated with 45.3% churn; month-to-month contracts at 42.7% churn — all surfaced without any manually predefined analytical rules.
- **Tech Stack**: LangGraph, Groq API (LLaMA 3.3 70B), LangChain, Pandas, Seaborn, Matplotlib

---

### 5. [Credit Card Fraud Detection System](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Credit%20Card%20Fraud%20Detection.ipynb)
- **Overview**: Developed a real-time fraud detection system using machine learning models including Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **Key Techniques**:
  - Extensive exploratory data analysis (EDA) to analyze transaction patterns and detect anomalies.
  - SMOTE for handling class imbalance.
  - Hyperparameter tuning to optimize model performance.
- **Outcome**: Developed a model with promising potential for fraud detection. While the model effectively reduces false positives, improving recall and overall fraud detection accuracy may be limited by the inherent class imbalance in the dataset.

---

### 6. [Stock Price Forecasting with LSTM](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Stock%20Price%20Analysis%20and%20Forecasting.ipynb)
- **Overview**: Created a stock price prediction model using Long Short-Term Memory (LSTM) networks with historical market data.
- **Key Techniques**:
  - Time-series analysis to uncover trends and seasonality in stock prices.
  - Financial modeling and correlation analysis between major tech companies.
- **Outcome**: Developed a forecasting model that generates actionable insights for investment strategies and price volatility predictions.

---

### 7. [Image Captioning with Deep Learning](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Image%20Captioning%20System.ipynb)
- **Overview**: Built an image captioning model using VGG16 for feature extraction and LSTM for natural language generation.
- **Key Techniques**:
  - EDA on the Flickr8k dataset to analyze image-text relationships.
  - Tokenization, sequence padding, and vocabulary management for generating captions.
- **Outcome**: Developed a scalable image captioning system for generating accurate descriptions for unseen images.

---

### 8. [Movie Recommendation System](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Movie%20Recommendation%20System.ipynb)
- **Overview**: Built a recommendation system using demographic, content-based, and collaborative filtering techniques with TMDB's dataset.
- **Key Techniques**:
  - EDA to identify key features such as genre, cast, and user ratings.
  - Integrated weighted rating formulas and collaborative filtering for personalized recommendations.
- **Outcome**: Delivered a scalable solution for personalized movie recommendations for streaming platforms.

---

### 9. [Sentiment Analysis of E-Commerce Reviews](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Sentiment%20Analysis%20of%20E-Commerce%20Clothing%20Reviews.ipynb)
- **Overview**: Developed a sentiment analysis model using Naive Bayes algorithms (Bernoulli and Multinomial) to classify customer reviews from an e-commerce platform. Additionally built models to predict customer feedback using Logistic Regression, KNN, Decision Trees, Random Forest, SVC, Naive Bayes, and XGBoost.
- **Key Techniques**:
  - EDA to analyze review patterns, identify sentiment trends, and visualize feature distributions.
  - Data preprocessing to handle missing values, outliers, and skewness; feature engineering to improve model accuracy.
  - Word clouds to visualize frequent terms and enhance feature understanding.
- **Outcome**: Provided actionable insights for product recommendations and customer satisfaction improvements.

---

### 10. [Customer Churn Analysis and Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Customer%20Churn%20Analysis%20and%20Prediction.ipynb)
- **Overview**: Analyzed customer churn in the telecom industry using the Telco customer dataset to predict churn and identify factors influencing retention. Built and evaluated several machine learning models.
- **Key Techniques**:
  - EDA to visualize churn patterns and feature distributions.
  - Data preprocessing: handled missing values, encoded categorical variables; feature selection via SelectKBest.
  - Built and compared models: Logistic Regression, Decision Trees, Random Forest, SVM, XGBoost, Naive Bayes.
  - Hyperparameter tuning using Grid Search for Random Forest.
- **Outcome**: Logistic Regression achieved the highest accuracy (80.01%) and F1 score (0.6053). Key insights highlighted the importance of tenure and contract type in predicting churn.

---

### 11. [Customer Segmentation using KMeans Clustering](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Customer%20Segmentation%20using%20KMeans%20Clustering.ipynb)
- **Overview**: Applied k-Means clustering to segment customers based on demographic and spending behaviors to uncover patterns and inform targeted marketing strategies.
- **Key Techniques**:
  - Data preprocessing: handled missing values and encoded categorical variables.
  - EDA to identify correlations and visualize data distributions.
  - Feature scaling using StandardScaler; optimal K=5 determined via the Elbow method.
  - Statistical tests: ANOVA and Chi-squared tests to analyze relationships between clusters and customer attributes.
- **Outcome**: Identified five distinct customer segments ranging from high spenders to low-income savers, providing insights for personalized marketing strategies.

---

### 12. [Clustering Countries by Socio-Economic and Health Indicators](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Clustering%20Countries%20by%20Development%20Indicators%20using%20Hierarchical%20Clustering.ipynb)
- **Overview**: Applied hierarchical clustering to categorize countries into Developed, Developing, and Least Developed clusters based on socio-economic and health indicators.
- **Key Techniques**:
  - Dendrogram analysis, Elbow Method, and Silhouette Score to determine optimal cluster count (3 clusters).
  - Agglomerative Hierarchical Clustering with StandardScaler normalization.
  - Visualized clusters using scatter plots across features: income vs GDP, life expectancy vs child mortality.
- **Outcome**: Successfully categorized countries into three development tiers. Findings can inform global development strategies and prioritize policy interventions.

---

### 13. [Life Expectancy (WHO) Analysis and Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Life%20Expectancy%20(WHO)%20Analysis%20and%20Prediction%20-%20Classification%20and%20Regression%20Models.ipynb)
- **Overview**: Analyzes and predicts life expectancy across countries using both regression and classification models, focusing on identifying influencing factors and classifying countries as developed or developing.
- **Key Techniques**:
  - Regression models: Linear Regression, Decision Trees, KNN, Random Forest, Gradient Boosting, AdaBoost, XGBoost, SVR.
  - Classification models: Logistic Regression, KNN, Decision Trees, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Gaussian Naive Bayes, SVC.
  - Feature selection via SelectKBest with ANOVA F-test; model tuning via GridSearchCV.
- **Outcome**: XGBoost was the best model for both regression and classification after hyperparameter tuning. Key influencing factors identified: Adult Mortality, Income Composition of Resources, and Schooling.

---

### 14. [Handwritten Digit Recognition with CNN](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Handwritten%20Digit%20Recognition%20with%20CNN.ipynb)
- **Overview**: Built a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset (60,000 training / 10,000 test images).
- **Key Techniques**:
  - PCA for 2D visualization of digit clustering; pixel normalization and one-hot encoding.
  - CNN with two convolutional layers, max-pooling, and a dense output layer.
  - Trained for 10 epochs using Adam optimizer and categorical crossentropy loss.
- **Outcome**: Achieved high test accuracy. CNNs demonstrated strong performance for image classification with proper preprocessing.

---

### 15. [Crowdfunding Success Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Crowdfunding%20Campaign%20Success%20Prediction.ipynb)
- **Overview**: Analyzed crowdfunding campaign success using machine learning models to predict project outcomes based on goal amount, category, and duration.
- **Key Techniques**:
  - EDA to explore relationships between features and campaign success.
  - Feature selection using Recursive Feature Elimination (RFE).
  - Built and compared models: Random Forest, Naive Bayes, SVM, XGBoost.
- **Outcome**: SVM achieved the highest accuracy (99.87%), with XGBoost (99.27%) recommended for better efficiency.

---

### 16. [Marketing Campaign Performance Insights](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Marketing%20Campaign%20Performance%20Insights.ipynb)
- **Overview**: Analyzed digital marketing campaigns to optimize ROI by examining conversion rates, acquisition costs, and engagement across channels, campaign types, and audience segments.
- **Key Techniques**:
  - Correlation analysis between metrics (clicks, ROI); customer segmentation by high-performing attributes.
  - Geographic and time-based insights to optimize marketing by location and seasonal trends.
- **Outcome**: Social media and influencer campaigns showed strongest ROI. Non-English segments (Spanish, Mandarin) had higher conversion rates. 60-day campaigns outperformed 30-day on ROI. Actionable insights on acquisition cost reduction and localization.

---

### 17. [Mobile Price Prediction using Linear Regression](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Mobile%20Price%20Prediction.ipynb)
- **Overview**: Predicted mobile prices based on features like RAM, camera quality, and battery using Linear Regression.
- **Key Techniques**:
  - EDA: analyzed feature distributions, correlations, and key price-influencing factors.
  - Data preprocessing: treated categorical variables and handled outliers.
  - Evaluated with R², MAE, MSE, and RMSE.
- **Outcome**: Model explained 85.8% of price variance (R² = 0.858, MAE = 189). Expanding feature set improved R² to 0.93.

---

### 18. [Campus Placement Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Campus%20Placement%20Prediction.ipynb)
- **Overview**: Developed and evaluated predictive models to classify students' placement status based on academic and extracurricular factors.
- **Key Techniques**:
  - Feature selection via correlation matrix and SelectKBest; Label Encoding for categorical features.
  - Built and evaluated Logistic Regression, Decision Tree, and KNN models.
- **Outcome**: Logistic Regression performed best at 79.8% accuracy. KNN (k=15) at 79.1%; Decision Tree (max_depth=3) at 77.6%.

---

### 19. [Diabetes Diagnosis Based on Patient Health Metrics](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Diabetes%20Diagnosis%20Based%20on%20Patient%20Health%20Metrics.ipynb)
- **Overview**: Predicted diabetes diagnosis (Diabetic, Non-Diabetic, Pre-diabetic) based on patient health metrics including age, BMI, HbA1c, and cholesterol levels.
- **Key Techniques**:
  - EDA to visualize feature distributions and correlations; SMOTE for class imbalance.
  - Built and evaluated: Logistic Regression, Random Forest, KNN, SVM.
- **Outcome**: Random Forest achieved the highest accuracy (98.15%) and F1-score (98.16%), recommended for deployment.

---

### 20. [Bank Loan Performance Analysis — Power BI Dashboard](https://app.powerbi.com/view?r=eyJrIjoiODgwOTQ2ZWQtMDEyMy00MjAyLThjMDQtOTE1ZGRhZDcyMGUzIiwidCI6Ijk0NWFlNmVkLWJiZmYtNGM0My05YjRhLWZkNDJmMDRiY2FkZSJ9&authuser=0)
- **Overview**: Analyzed a loan dataset to understand how borrower details and loan characteristics affect loan performance, with the goal of optimizing lending strategies and mitigating credit risk.
- **Key Techniques**:
  - Data transformation: cleaned and formatted data, created computed columns (total_amount_paid, delinquency_status, remaining_installments).
  - DAX measures for KPIs like Fully Paid Loan Percentage; dynamic visuals for loan performance and borrower profiles.
- **Outcome**: Delivered actionable insights on loan trends by amount, status, and interest rate; borrower demographic and delinquency analysis for improved lending strategy.

---

### 21. [Ecomm Sales Optimization & Customer Satisfaction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Ecomm%20Sales%20Optimization%20%26%20Customer%20Satisfaction.ipynb)
- **Overview**: Analyzed e-commerce sales data to optimize strategies and improve customer satisfaction by exploring browsing, sales, customer, and feedback datasets.
- **Key Techniques**:
  - Data merging (inner, left join); missing value handling (interpolation, mean/mode imputation); cohort analysis and sentiment analysis.
  - Random Forest Regressor to predict Sales_Amount; K-Means Clustering for customer segmentation.
  - Sales and feedback dashboards for trend visualization.
- **Outcome**: Identified top-performing regions and products; built accurate sales forecasting model; created customer segments for targeted marketing.

---

### 22. [Loan Approval Prediction and Analysis](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Loan%20Approval%20Prediction%20and%20Analysis.ipynb)
- **Overview**: Predicts loan approval outcomes using data analysis and machine learning, exploring features like applicant income, credit history, and loan amount. Focused on F1 score to handle class imbalance.
- **Key Techniques**:
  - KNN, mode, and median imputation for missing values; IQR-based outlier removal; square root transformation for skewness.
  - Applied classifiers: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, KNN.
- **Outcome**: Logistic Regression achieved the best performance balancing precision and recall. Credit history and income identified as key approval factors.

---

### 23. [Data-Driven Insights into Realtime Real Estate Price Prediction](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/Data%20Driven%20Insights%20into%20Realtime%20Real%20Estate%20Price%20Prediction.ipynb)
- **Overview**: Developed a machine learning model to predict real estate prices based on property features including title, location, carpet area, and amenities, with end-to-end preprocessing and feature engineering.
- **Key Techniques**:
  - Area unit standardization to square feet; IQR-based outlier removal; square-root transformation for skewness.
  - Word clouds for location frequency analysis; multiple regression models evaluated.
- **Outcome**: Cleaned dataset improved prediction accuracy. Visualizations revealed trends such as most expensive locations and impact of amenities on pricing.

---

### 24. [Bank Customer Churn Analysis (SQL)](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/BankCustomerChurn_Analysis.sql)
- **Overview**: Examined customer churn in a banking dataset using SQL, identifying patterns in customer behavior, demographics, and financial attributes that influence churn.
- **Key Techniques**:
  - SQL aggregations across demographics, financial metrics, and behavioral patterns.
  - Customer segmentation by churn status, age group, tenure range, and credit score.
  - Derived churn rates by country, tenure, and product usage.
- **Outcome**: Identified key churn drivers including age, credit score, and activity level. Provided actionable metrics for targeted retention interventions.

---

### 25. [E-Commerce Customer Churn Data Wrangling & Analysis (SQL)](https://github.com/ArchanaAnalytics/DS-Portfolio/blob/main/Projects/E-Commerce%20Customer%20churn%20Analysis.sql)
- **Overview**: Focused on cleaning and analyzing customer churn data by handling missing values, removing outliers, and transforming data to derive insights into customer behaviors, demographics, and churn patterns.
- **Key Techniques**:
  - Missing value imputation, outlier removal, and data standardization.
  - Column transformation: renamed columns, added ChurnStatus, dropped irrelevant fields.
  - Aggregated data to uncover churn patterns by device, payment method, and city tier.
- **Outcome**: Generated insights on churned customer demographics, preferred devices, and payment methods. Provided actionable metrics including complaint rates among churned customers and city-tier churn rates.

---

## Skills & Tools

| Category | Tools & Technologies |
|---|---|
| **Generative AI** | RAG Pipelines, LangChain, FAISS, RAGAS, Prompt Engineering |
| **Agentic AI** | LangGraph, CrewAI, Multi-Agent Orchestration, Tool Use |
| **LLMs** | Mistral-7B, LLaMA 3.3 70B (Groq), OpenAI GPT-4o-mini, HuggingFace |
| **Machine Learning** | Scikit-learn, XGBoost, Random Forest, SVM, Logistic Regression |
| **Deep Learning** | TensorFlow, Keras, CNN, LSTM, VGG16 |
| **NLP** | Sentiment Analysis, Text Classification, Tokenization, Word Clouds |
| **Data Analysis** | Pandas, NumPy, EDA, Feature Engineering, Statistical Testing |
| **Visualization** | Matplotlib, Seaborn, Power BI |
| **Backend / APIs** | FastAPI, Python AsyncIO, Pydantic |
| **Databases / Query** | SQL (MySQL / SQLite) |
| **Other** | SMOTE, PCA, Clustering, Hyperparameter Tuning, GridSearchCV |
