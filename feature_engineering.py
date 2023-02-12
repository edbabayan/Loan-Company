import pandas as pd
import numpy as np

df = pd.read_csv('lending_club_loan_two.csv')

print('Create a new column for main label with numbers.')
df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
print('\n')

print("Next step is to remove or to fill the missing data.")
length = len(df)
print(f"The length of the dataframe is {length}")
print('\n')

print(df.isnull().sum())
print('\n')

print(df['emp_title'].nunique())
print("""As we can see the number of job titles is too high,
so we can't use get_dummies on it, it will be better just drop that column.""")
print('\n')
df.drop('emp_title', axis=1, inplace=True)

emp_co = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()['loan_status']
emp_fp = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length').count()['loan_status']
print(emp_co/emp_fp)
print('\n')

print("As the rates are extremely similar across all employment lengths, drop that column.")
df.drop('emp_length', axis=1, inplace=True)
print('\n')

print("The columns title and purpose are nearly the same, so we can drop one of them.")
df.drop('title', axis=1, inplace=True)
print('\n')

print('Now is the time for the hardest step.')
print("Fill the mortgage account using the total accounts.")

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(total_acc) or np.isnan(mort_acc):
        return 0
    elif np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

print("So we have ended filling the missing values.")
print("""As the missing values of other columns are not to big,
we can just drop that rows.""")

df.dropna(inplace=True)
