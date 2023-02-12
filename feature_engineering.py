import pandas as pd

df = pd.read_csv('lending_club_loan_two.csv')

print('Create a new column for main label with numbers.')
df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
print('\n')

print("Next step is to remove or to fill the missing data.")
length = len(df)
print(f"The length of the dataframe is {length}")
print('\n')

print(df.isnull().sum())
