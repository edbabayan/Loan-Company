import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('lending_club_loan_two.csv')

print("Let's create a count plot to explore the balancing of our label.")
# sns.countplot(x='loan_status', data=df)
print('\n')

print("Next lets make a histogram of loan amount.")
# sns.displot(x='loan_amnt', data=df)
print('\n')

print("Explore the correlation between the continuous feature variables.")
# sns.heatmap(df.corr())
print('\n')

print('Create a boxplot showing the relationship between the loan status and loan amount.')
# sns.boxplot(x='loan_status', y='loan_amnt', data=df)
print('\n')

print('Create a count plot per grade.')
grade_order = sorted(df['grade'].unique())
sns.countplot(x='grade', data=df, hue='loan_status', order=grade_order)
plt.savefig('grade.jpg')

