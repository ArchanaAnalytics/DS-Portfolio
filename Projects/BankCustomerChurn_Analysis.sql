#		BANK CUSTOMER CHURN ANALYSIS

USE bank_customers;

# Total Number of Customers by Gender
SELECT gender, COUNT(*) AS Customer_Count FROM customer_churn GROUP BY gender;

# Average Age of Customers Who Churned
SELECT FLOOR(AVG(age)) AS Avg_Age FROM customer_churn WHERE churn=1;

# Percentage of Customers Who Churned
SELECT IF(churn=1, 'Yes', 'No') AS Churn, 
	CONCAT(ROUND(COUNT(*)/(SELECT COUNT(*) FROM customer_churn)*100, 2), '%') AS Churn_Percentage
	FROM customer_churn
	GROUP BY churn;

# Average Credit Score of Customers Who Churned vs. Stayed
SELECT IF(churn=1, 'Yes', 'No') AS Churn, ROUND(AVG(credit_score)) AS Avg_Credit_Score
	FROM customer_churn
	GROUP BY churn;

# Churn Rate by Country
SELECT country, COUNT(*) / (SELECT COUNT(*) FROM customer_churn) * 100 AS Churn_Rate
	FROM customer_churn
	WHERE churn=1
	GROUP BY country;

# Average Balance by Tenure
SELECT tenure, ROUND(AVG(balance),2) AS Avg_Balance
	FROM customer_churn
	GROUP BY tenure
    ORDER BY tenure;

# Churn Rate Among Customers with Credit Cards
SELECT IF(credit_card=1, 'Yes', 'No') AS HasCreditCard, COUNT(*) / (SELECT COUNT(*) FROM customer_churn WHERE credit_card = 1) * 100 AS Churn_Rate
	FROM customer_churn
	WHERE churn=1
	GROUP BY credit_card;

  # Churned Customers Who Are Active Members And Has Credit Card
SELECT COUNT(*) AS Churned_ActiveMembers_With_CreditCard
FROM customer_churn
WHERE churn = 1 AND credit_card = 1 AND active_member = 1;

# Churned Customers with High Credit Score
SELECT COUNT(*) AS HighCreditScore_ChurnedCustomers
	FROM customer_churn
	WHERE churn=1 AND credit_score > 800;

# Customers Who Are Active Members vs. Inactive
SELECT IF(active_member=1, 'Yes', 'No') AS Active_Members, COUNT(*) AS Customer_Count
FROM customer_churn
GROUP BY active_member;

# Gender Wise Inactive Members Who Churned
SELECT gender, COUNT(*) AS Inactive_Customer_Count
	FROM customer_churn
    WHERE active_member=0 AND churn=1
    GROUP BY gender;

# Average Tenure of Customers by Churn Status
SELECT IF(churn=1, 'Yes', 'No') AS Churn, ROUND(AVG(tenure)) AS Avg_Tenure
	FROM customer_churn
	GROUP BY churn;

# Average Salary of Customers
SELECT IF(churn=1, 'YES', 'NO') AS Churn, ROUND(AVG(estimated_salary),2) AS Avg_Salary
	FROM customer_churn
	GROUP BY churn;

# Churned Customer-IDs with their average salaries greater than the overall Average Salary of all the Customers
SELECT customer_id, ROUND(AVG(estimated_salary),2) AS Avg_Salary
	FROM customer_churn
	WHERE churn=1
	GROUP BY customer_id
	HAVING Avg_Salary > (SELECT AVG(estimated_salary) FROM customer_churn)
    ORDER BY Avg_Salary DESC; 
    
# Average Products Count for Churned Customers:
SELECT ROUND(AVG(products_number)) AS Products_Count
	FROM customer_churn
	WHERE churn=1;

# Average Age of Churned Customers by Country
SELECT country, ROUND(AVG(age)) AS Avg_Age
	FROM customer_churn
	WHERE churn=1
	GROUP BY country;
    
# Customers Who Has Zero Balance
SELECT IF(churn=1, 'YES', 'NO') AS Churn, COUNT(*) AS Customers_With_ZeroBalance
	FROM customer_churn
    WHERE balance=0
    GROUP BY Churn;

# Customer Churn by Age Group
SELECT
    CASE
        WHEN Age < 30 THEN '18-30'
        WHEN Age BETWEEN 30 AND 39 THEN '30-39'
        WHEN Age BETWEEN 40 AND 49 THEN '40-49'
        WHEN Age BETWEEN 50 AND 59 THEN '50-59'
        ELSE '60+'
    END AS Age_Group,
    COUNT(*) AS Churned_Customers
FROM customer_churn
WHERE churn=1
GROUP BY Age_Group
ORDER BY Age_Group;

# Customer Churn by Tenure Range
SELECT
    CASE
        WHEN Tenure < 12 THEN '0-11 months'
        WHEN Tenure BETWEEN 12 AND 24 THEN '12-24 months'
        WHEN Tenure BETWEEN 25 AND 36 THEN '25-36 months'
        ELSE '37+ months'
    END AS Tenure_Range,
    COUNT(*) AS Churned_Customers
FROM customer_churn
WHERE churn=1
GROUP BY Tenure_Range
ORDER BY Tenure_Range;

# CTE to Calculate the average credit score of customers who have churned, grouped by gender and country. 
-- And from the CTE, identify the countries with the highest and lowest average credit scores among churned customers:
WITH Churned_Customers AS (
    SELECT gender, country, AVG(credit_score) AS Avg_CreditScore
    FROM customer_churn
    WHERE churn=1
    GROUP BY gender, country)

SELECT country, gender, Avg_CreditScore,
    CASE
        WHEN Avg_CreditScore = (SELECT MAX(Avg_CreditScore) FROM Churned_Customers) THEN 'Highest'
        WHEN Avg_CreditScore = (SELECT MIN(Avg_CreditScore) FROM Churned_Customers) THEN 'Lowest'
        ELSE 'Other'
    END AS CreditScore_Category
FROM Churned_Customers
ORDER BY country, gender;

