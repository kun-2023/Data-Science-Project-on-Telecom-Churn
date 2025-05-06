# Data Science Project on Telecom Customers Churns with Classification and Clustering Models 

## Case Descriptions
The project is to analyze customer churn data from a telecom company. The goal is to find patterns in clients who has churned, build a predictive machine learning model for forecasting churns, and cluster customers by an unsupervised model so the company can tailor each cluster for a better service and a lower churn rate.
## Data Source
Data Science Bootcamp. 
## Technology
Pandas, Python3, Numpy, Seaborn, Matplotlib, Sklearn, Tensorflow
## Data Science Methods
Data Manipulation, Data Visualization, GridSearchCV, LogisticRegression, DecisionTree, Keras, PCA
## Data Description
The dataset has 7043 rows and 33 features. There are null values in churn reason columns and Total Charges.
![image](https://github.com/user-attachments/assets/1c342b4c-3208-4e1b-9de3-0dd7733a3611)

 
## Data Cleaning
Fill in those null values in Churn Reason with “Missing”.  Convert total charges column from object to numeric and fill in null values with the mean value of total charges.
Convert binary categorical columns into numeric features with 0s and 1s. 

## DEA Findings
1.	Seniors count for small portions of the total customers. Many clients don’t have dependents. Majority of the clients have total charges of under 1000 dollars. Many clients have tenure months of either under 10 months or over 60 months.

![image](https://github.com/user-attachments/assets/2724a6e8-59ab-4182-bb20-69e30953eeea)



2.	There are 7043 clients or data points in total. 1869 churned and 5174 clients had stayed. Thus, the churn rate is 26.54%.
3.	Churn Rate for different categories. Clients with partners or dependents have low churn rate. Clients with fiber optic have a higher churn rate. Customers with no online securities have a high churn rate. For most of the features, majority of clients with no internet services at all have decided to stay with the company. 
![image](https://github.com/user-attachments/assets/c022d48a-c624-4ab9-b1f6-61cef8d89449)


 
4.	The top 5 reasons that customers had left the company are attitude of the support person, better speed from competitors, more data from competitors, don’t know, and better offer from competitors. The top 5 reasons count for 42.05% of the people who churned. 

5.	When tenure month gets longer, people are more likely to stay. Short term contract clients are very likely to churn. Consequentially, people with higher total charges tend to stay the company. However, clients with a high monthly payment are very likely to churn. 

![image](https://github.com/user-attachments/assets/d391ac06-a2b6-4ab2-b617-1983a357818b)



 
6. Few features are correlated. Partners correlated with dependents, tenure months, and total charges. Total charges strongly correlated with tenure months and monthly charges. Churn Value negatively correlated with dependent, partner, and total charges. 
![image](https://github.com/user-attachments/assets/6abb691b-2239-4f7e-8e51-52282c7ce338)

7. Location of churned and unchurned customers are all cross California, so there are no pattern in regards to the address of the churned and unchurned customers.


![image](https://github.com/user-attachments/assets/3ab04682-ba00-477f-bf41-a9db3bc88682)


## Predictive Model – Logistic Regression
8.	Feature Engineering. Drop non-related columns such as Churn Reason, Churn Value, City. Perform one hot encoding on all categorical features.
9.	Model tuning. Trained model with grid search. C set to be any number between 0.1 and 2.1 with 0.1 apart from each other. Penalty set as l1 and l2. The best parameters are C=0.1 and penalty as l1.
10.	Model evaluation. The training accuracy was 75% and the testing accuracy was 74%. The precision for negative cases was 90%, and the recall was 73%. For positive cases, the precision was 54%, and the recall was 81%. The model was not overfit, and the accuracy was acceptable. However, the model was good at predicting negative cases, but not so for positive cases.


![image](https://github.com/user-attachments/assets/d17ce61b-4101-4223-8268-052853930071)


 
## Predictive Model – Decision Tree
11.	The best parameters are {'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': np.int64(15)}. The training accuracy was 79%, the testing accuracy was 72%. It’s overfit. Retrain the model with the most important features.
12.	Retrain decision tree model with only contract, tenure months, internet service and dependent. The best parameters are {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': np.int64(15)}. 
13.	Model Evaluation. The training accuracy was 77%, and the testing accuracy was 73%. For the negative cases, the precision was 89%, and the recall was 72%. For the positive cases, the precision was 52%, and the recall was 79%. The model was overfit, and the model was better at predicting the negative cases than positive cases. 
 ![image](https://github.com/user-attachments/assets/cd323a61-eab0-4bad-92c1-2d9be6b2e184)
 

## Predictive Model – Deep Learning Binary
14. I set three hidden layers with 10 neurons for each. To reduce overfitting, I set the dropout rate as 20% for the first layer.

![image](https://github.com/user-attachments/assets/16e951bd-c30a-4822-9fcf-2bc618a5fdfa)



15.	 Model had achieved a much higher accuracy than both previous models. The training accuracy was 82% and the testing accuracy was 80%. For the negative cases, the precision was 83%, and the recall was 91%. For the positive cases, the precision was 70%, and the recall was 53%. This model was not overfit with a high accuracy. Again, the model is better at predicting negative cases than positive. It’s the best model we have.
 
 ![image](https://github.com/user-attachments/assets/681bb607-01a4-4297-b761-ba757cace093)



## Cluster Model – K Means Clustering
16.	The best cluster is 3 based on the silhouette scores and inertias. The three clusters are set as the following:
•	Cluster 0: Most have Phone Service, Internet with less Internet Services. Medium Charges.
•	Cluster 1: Everyone has phone services, No Internet Services. Low Charges.
•	Cluster 2: Most have phone services. Internet with lots of Internet Servies. High Charges.
Cluster 0 has the highest churn rate. Probably they bough services they don’t need. Cluster 1 has the lowest churn rate. They don’t have internet. They are having what they need. Cluster 3 has the phone, internet and lots of online services. Most of them have partners. They have busy life and they can share the cost with their partners.



![image](https://github.com/user-attachments/assets/ff98f598-54c9-4da8-aa9a-5f0223cc5233)

![image](https://github.com/user-attachments/assets/efeb511a-0356-4cc7-93ad-03dab7b35d6b)

 
17.	PCA visualization of the clusters. Set PCA 2 components. The first 2 components cover 42% of the explained variances. The three clusters had been clustered clearly.


![image](https://github.com/user-attachments/assets/0bc9b1b6-67ed-4efb-b4c3-9df7a8ed2f81)


 

## Conclusion & Recommendations
*	Clients with short term contracts are not likely stick around. Seniors have a high churn rate. Clients without internet connection and only a phone services tend to stick to the company.
*	For Cluster 0 with the highest churn rate, not all of them need phones or internet. Talk to them and only offer them what they need.
*	For Cluster 1 with lowest churn rate. They only have phone services, but they have everything they need, and they are staying with the company. Call them see what else they need.
*	For cluster 2 with medium level of churn rate. They have lots of internet related services, and they are paying a lot, and they are staying. Promote new value added internet services to them every time when there is a new product coming out.

  ## The End

