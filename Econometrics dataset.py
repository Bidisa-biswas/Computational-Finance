import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/Users/bidisabiswas/Library/Mobile Documents/com~apple~CloudDocs/Master course/Econometrics 1/Dataset/Participation.csv')
data.head()  # Display the first 5 rows of the dataset

# Set random seed for reproducibility of the results
np.random.seed(100)

# Generate the synthetic data
data['education'] = np.random.choice(['high', 'medium', 'low'], size=len(data))
data['industry'] = np.random.choice(['A', 'B', 'C'], size=len(data))

# Map categorical data to numerical values
education_mapping = {'low': 1, 'medium': 2, 'high': 3}
industry_mapping = {'A': 1, 'B': 2, 'C': 3}
data['education'] = data['education'].map(education_mapping)
data['industry'] = data['industry'].map(industry_mapping)

# Calculate salary
data['salary'] = 1000 + 2000 * data['education'] + 3000 * data['industry'] + np.random.normal(0, 1000, len(data))

# Calculate and display the mean salary for each industry
mean_salary_by_industry = data.groupby('industry')['salary'].mean()
print(mean_salary_by_industry)

# Calculate the mean salary for the education level
mean_salary_by_education = data.groupby('education')['salary'].mean()
print(mean_salary_by_education)

# Calculate the mean salary for the education and the industry together
mean_education_industry_salary = data.groupby(['industry', 'education'])['salary'].mean()
print(mean_education_industry_salary)

# Find the correlation between the salary and the education level
correlation_education = data['salary'].corr(data['education'])
print(correlation_education)

# Find the correlation between the salary and the industry
correlation_industry = data['salary'].corr(data['industry'])
print(correlation_industry)

# Plot the salary distribution for each industry
sns.boxplot(x='industry', y='salary', data=data)
plt.show()
