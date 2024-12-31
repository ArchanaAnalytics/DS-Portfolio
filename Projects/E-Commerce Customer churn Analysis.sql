USE ecomm;

-- Data Cleaning:
SET SQL_SAFE_UPDATES = 0;

-- Handling Missing Values and Outliers

-- Calculate rounded mean values using user-defined variables
SET @WarehouseToHome_avg = (SELECT ROUND(AVG(WarehouseToHome)) FROM customer_churn);
SET @HourSpendOnApp_avg = (SELECT ROUND(AVG(HourSpendOnApp)) FROM customer_churn);
SET @OrderAmountHikeFromlastYear_avg = (SELECT ROUND(AVG(OrderAmountHikeFromlastYear)) FROM customer_churn);
SET @DaySinceLastOrder_avg = (SELECT ROUND(AVG(DaySinceLastOrder)) FROM customer_churn);

-- Impute the calcuated mean for the specified columns
UPDATE customer_churn
SET WarehouseToHome = IF(WarehouseToHome IS NULL, @WarehouseToHome_avg, WarehouseToHome),
    HourSpendOnApp = IF(HourSpendOnApp IS NULL, @HourSpendOnApp_avg, HourSpendOnApp),
    OrderAmountHikeFromlastYear = IF(OrderAmountHikeFromlastYear IS NULL, @OrderAmountHikeFromlastYear_avg, OrderAmountHikeFromlastYear),
    DaySinceLastOrder = IF(DaySinceLastOrder IS NULL, @DaySinceLastOrder_avg, DaySinceLastOrder);

-- Calculate mode using user-defined variables
SET @Tenure_mode = (SELECT Tenure FROM customer_churn GROUP BY Tenure ORDER BY COUNT(*) DESC LIMIT 1);
SET @CouponUsed_mode = (SELECT CouponUsed FROM customer_churn GROUP BY CouponUsed ORDER BY COUNT(*) DESC LIMIT 1);
SET @OrderCount_mode = (SELECT OrderCount FROM customer_churn GROUP BY OrderCount ORDER BY COUNT(*) DESC LIMIT 1);

-- Impute mode for the specified columns
UPDATE customer_churn
SET Tenure = IF(Tenure IS NULL, @Tenure_mode, Tenure),
    CouponUsed = IF(CouponUsed IS NULL, @CouponUsed_mode, CouponUsed),
    OrderCount = IF(OrderCount IS NULL, @OrderCount_mode, OrderCount);

-- Handle outliers in 'WarehouseToHome' column
DELETE FROM customer_churn
WHERE WarehouseToHome > 100;


-- Dealing with Inconsistencies

-- Replace occurrences of "Phone" or “Mobile” with "Mobile Phone" in the specified columns
UPDATE customer_churn
SET PreferredLoginDevice = IF(PreferredLoginDevice = 'Phone', 'Mobile Phone', PreferredLoginDevice),
    PreferedOrderCat = IF(PreferedOrderCat = 'Mobile', 'Mobile Phone', PreferedOrderCat);

-- Standardize payment mode values
UPDATE customer_churn
SET PreferredPaymentMode = CASE 
                                WHEN PreferredPaymentMode = 'COD' THEN 'Cash on Delivery'
                                WHEN PreferredPaymentMode = 'CC' THEN 'Credit Card'
                                ELSE PreferredPaymentMode
                           END;


-- Data Transformation:

-- Column Renaming
ALTER TABLE customer_churn
RENAME COLUMN PreferedOrderCat TO PreferredOrderCat,
RENAME COLUMN HourSpendOnApp TO HoursSpentOnApp;

-- Creating New Columns
ALTER TABLE customer_churn
ADD COLUMN ComplaintReceived ENUM('Yes','No'),
ADD COLUMN ChurnStatus ENUM('Churned','Active');

-- Set values for the new columns based on existing data
UPDATE customer_churn
SET 
    ComplaintReceived = IF(Complain = 1, 'Yes', 'No'),
    ChurnStatus = IF(Churn = 1, 'Churned', 'Active');

-- Column Dropping
ALTER TABLE customer_churn
DROP COLUMN Churn,
DROP COLUMN Complain;


-- Data Exploration and Analysis:
SELECT * FROM customer_churn;
SELECT DISTINCT PreferredLoginDevice FROM customer_churn;
SELECT DISTINCT PreferredPaymentMode FROM customer_churn;
SELECT DISTINCT PreferredOrderCat FROM customer_churn;
SELECT CustomerID FROM customer_churn WHERE WarehouseToHome IS NULL;
SELECT CustomerID FROM customer_churn WHERE OrderCount IS NULL;


USE ecomm;
SELECT * FROM customer_churn;
SELECT DISTINCT PreferredLoginDevice FROM customer_churn;
SELECT DISTINCT PreferredPaymentMode FROM customer_churn;
SELECT DISTINCT PreferredOrderCat FROM customer_churn;
SELECT CustomerID FROM customer_churn WHERE WarehouseToHome IS NULL;
SELECT CustomerID FROM customer_churn WHERE OrderCount IS NULL;

-- 1. Retrieve the count of churned and active customers from the dataset.
SELECT ChurnStatus, COUNT(*) AS CustomerCount
FROM customer_churn
GROUP BY ChurnStatus;

-- 2. Display the average tenure of customers who churned.
SELECT ROUND(AVG(Tenure)) AS AvgTenureChurned
FROM customer_churn
WHERE ChurnStatus = 'Churned';

-- 3. Calculate the total cashback amount earned by customers who churned.
SELECT SUM(CashbackAmount) AS TotalCashbackChurned
FROM customer_churn
WHERE ChurnStatus = 'Churned';

-- 4. Determine the percentage of churned customers who complained.
SELECT (COUNT(*) * 100 / (SELECT COUNT(*) FROM customer_churn WHERE ChurnStatus = 'Churned')) AS ChurnedComplaintPercentage
FROM customer_churn
WHERE ChurnStatus = 'Churned' AND ComplaintReceived = 'Yes';

-- 5. Find the gender distribution of customers who complained.
SELECT Gender, COUNT(*) AS ComplaintCount
FROM customer_churn
WHERE ComplaintReceived = 'Yes'
GROUP BY Gender;

-- 6. Identify the city tier with the highest number of churned customers whose prefered order category is Laptop & Accessory.
SELECT CityTier, COUNT(*) AS ChurnedCount
FROM customer_churn
WHERE ChurnStatus = 'Churned' AND PreferredOrderCat = 'Laptop & Accessory'
GROUP BY CityTier
ORDER BY ChurnedCount DESC
LIMIT 1;

-- 7. Identify the most preferred payment mode among active customers.
SELECT PreferredPaymentMode, COUNT(*) AS ActiveCount
FROM customer_churn
WHERE ChurnStatus = 'Active'
GROUP BY PreferredPaymentMode
ORDER BY ActiveCount DESC
LIMIT 1;

-- 8. List the preferred login device(s) among customers who took more than 10 days since their last order.
SELECT PreferredLoginDevice, COUNT(*) AS DeviceCount
FROM customer_churn
WHERE DaySinceLastOrder > 10
GROUP BY PreferredLoginDevice;

-- 9. List the number of active customers who spent more than 3 hours on the app.
SELECT COUNT(*) AS ActiveCustomers
FROM customer_churn
WHERE ChurnStatus = 'Active' AND HoursSpentOnApp > 3;

-- 10. Find the average cashback amount received by customers who spent atleast 2 hours on the app.
SELECT AVG(CashbackAmount) AS AvgCashbackForHours
FROM customer_churn
WHERE HoursSpentOnApp >= 2;

-- 11. Display the maximum hours spent on the app by customers in each preferred order category.
SELECT PreferredOrderCat, MAX(HoursSpentOnApp) AS MaxHoursSpent
FROM customer_churn
GROUP BY PreferredOrderCat;

-- 12. Find the average order amount hike from last year for customers in each marital status category.
SELECT MaritalStatus, AVG(OrderAmountHikeFromlastYear) AS AvgOrderAmountHike
FROM customer_churn
GROUP BY MaritalStatus;

-- 13. Calculate the total order amount hike from last year for customers who are single and prefer mobile phones for ordering.
SELECT SUM(OrderAmountHikeFromlastYear) AS TotalOrderAmountHike
FROM customer_churn
WHERE MaritalStatus = 'Single' AND PreferredOrderCat = 'Mobile Phone';

-- 14. Find the average number of devices registered among customers who used UPI as their preferred payment mode.
SELECT FLOOR(AVG(NumberOfDeviceRegistered)) AS AvgDevicesRegisteredUPI
FROM customer_churn
WHERE PreferredPaymentMode = 'UPI';

-- 15. Determine the city tier with the highest number of customers.
SELECT CityTier, COUNT(*) AS CustomerCount
FROM customer_churn
GROUP BY CityTier
ORDER BY CustomerCount DESC
LIMIT 1;

-- 16. Find the marital status of customers with the highest number of addresses.
SELECT MaritalStatus
FROM customer_churn
WHERE NumberOfAddress = (SELECT MAX(NumberOfAddress) FROM customer_churn);

-- 17. Identify the gender that utilized the highest number of coupons.
SELECT Gender, SUM(CouponUsed) AS TotalCouponsUsed
FROM customer_churn
GROUP BY Gender
ORDER BY TotalCouponsUsed DESC
LIMIT 1;

-- 18. List the average satisfaction score in each of the preferred order categories.
SELECT PreferredOrderCat, AVG(SatisfactionScore) AS AvgSatisfactionScore
FROM customer_churn
GROUP BY PreferredOrderCat;

-- 19. Calculate the total order count for customers who prefer using credit cards and have the maximum satisfaction score.
SELECT SUM(OrderCount) AS TotalOrderCount
FROM customer_churn
WHERE PreferredPaymentMode = 'Credit Card' AND SatisfactionScore = (SELECT MAX(SatisfactionScore) FROM customer_churn);

-- 20. How many customers are there who spent only one hour on the app and days since their last order was more than 5?
SELECT COUNT(*) AS CustomersCount
FROM customer_churn
WHERE HoursSpentOnApp = 1 AND DaySinceLastOrder > 5;

-- 21. What is the average satisfaction score of customers who have complained?
SELECT AVG(SatisfactionScore) AS AvgSatisfactionScoreComplained
FROM customer_churn
WHERE ComplaintReceived = 'Yes';

-- 22. How many customers are there in each preferred order category?
SELECT PreferredOrderCat, COUNT(*) AS CustomerCount
FROM customer_churn
GROUP BY PreferredOrderCat;

-- 23. What is the average cashback amount received by married customers?
SELECT AVG(CashbackAmount) AS AvgCashbackMarried
FROM customer_churn
WHERE MaritalStatus = 'Married';

-- 24. What is the average number of devices registered by customers who are not using Mobile Phone as their preferred login device?
SELECT FLOOR(AVG(NumberOfDeviceRegistered)) AS AvgDevicesRegisteredNonMobile
FROM customer_churn
WHERE PreferredLoginDevice != 'Mobile Phone';

-- 25. List the preferred order category among customers who used more than 5 coupons.
SELECT PreferredOrderCat, COUNT(*) AS CouponUsedCount
FROM customer_churn
WHERE CouponUsed > 5
GROUP BY PreferredOrderCat
ORDER BY CouponUsedCount DESC;

-- 26. List the top 3 preferred order categories with the highest average cashback amount.
SELECT PreferredOrderCat, AVG(CashbackAmount) AS AvgCashback
FROM customer_churn
GROUP BY PreferredOrderCat
ORDER BY AvgCashback DESC
LIMIT 3;

-- 27. Find the preferred payment modes of customers whose average tenure is 10 months and have placed more than 500 orders.
SELECT PreferredPaymentMode, ROUND(AVG(Tenure)) AS AvgTenure, COUNT(*) AS OrderCount
FROM customer_churn
GROUP BY PreferredPaymentMode
HAVING AvgTenure = 10 AND OrderCount>500;

-- 28. Categorize customers based on their distance from the warehouse to home such as 'Very Close Distance' for distances <=5km, 'Close Distance' for <=10km, 'Moderate Distance' for <=15km, and 'Far Distance' for >15km. Then, display the churn status breakdown for each distance category.
SELECT 
	ChurnStatus,
    CASE 
        WHEN WarehouseToHome <= 5 THEN 'Very Close Distance'
        WHEN WarehouseToHome <= 10 THEN 'Close Distance'
        WHEN WarehouseToHome <= 15 THEN 'Moderate Distance'
        ELSE 'Far Distance' 
    END AS DistanceCategory,
    COUNT(*) AS CustomerCount
FROM customer_churn
GROUP BY ChurnStatus, DistanceCategory
ORDER BY ChurnStatus, FIELD(DistanceCategory, 'Very Close Distance', 'Close Distance', 'Moderate Distance', 'Far Distance');

-- 29. List the customer’s order details who are married, live in City Tier-1, and their order counts are more than the average number of orders placed by all customers.
SELECT *
FROM customer_churn
WHERE MaritalStatus = 'Married' 
AND CityTier = 1 
AND OrderCount > (SELECT AVG(OrderCount) FROM customer_churn);


-- 30. a) Create a 'customer_returns' table in the 'ecomm' database and insert the provided data:
CREATE TABLE IF NOT EXISTS ecomm.customer_returns (
    ReturnID INT PRIMARY KEY,
    CustomerID INT,
    ReturnDate DATE,
    RefundAmount DECIMAL(10, 2)
);

-- Insert data into the 'customer_returns' table:
INSERT INTO ecomm.customer_returns (ReturnID, CustomerID, ReturnDate, RefundAmount)
VALUES
(1001, 50022, '2023-01-01', 2130.00),
(1002, 50316, '2023-01-23', 2000.00),
(1003, 51099, '2023-02-14', 2290.00),
(1004, 52321, '2023-03-08', 2510.00),
(1005, 52928, '2023-03-20', 3000.00),
(1006, 53749, '2023-04-17', 1740.00),
(1007, 54206, '2023-04-21', 3250.00),
(1008, 54838, '2023-04-30', 1990.00);

-- 30. b) Display the return details along with the customer details of those who have churned and have made complaints.
SELECT cr.*, cc.*
FROM customer_returns cr
JOIN customer_churn cc ON cr.CustomerID = cc.CustomerID
WHERE cc.ChurnStatus = 'Churned' AND cc.ComplaintReceived = 'Yes';

