import warnings
import pandas as pd
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('lending_club_loan_two.csv')
df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
df.drop('emp_title', axis=1, inplace=True)
df.drop('emp_length', axis=1, inplace=True)
df.drop('title', axis=1, inplace=True)

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(total_acc) or np.isnan(mort_acc):
        return 0
    elif np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
df.dropna(inplace=True)

print("In this file we are gonna look for object columns.")
print(df.dtypes)
print('\n')

df['term'] = df['term'].map({"36 months": 36, "60 months": 60})

print("As the saw in previous file columns grade and subgrade are also similar.")
df.drop('grade', axis=1, inplace=True)
print('\n')

print("Let's create dummy variables for subgrade column.")
print('\n')
dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis=1)

print("Do the same for other object columns.")
print('\n')

df['verification_status'] = df['verification_status'].astype('category')
df['application_type'] = df['application_type'].astype('category')
df['initial_list_status'] = df['initial_list_status'].astype('category')
df['purpose'] = df['purpose'].astype('category')


dummies = pd.get_dummies(df[['verification_status', 'application_type',
                         'initial_list_status', 'purpose']], drop_first=True)
df = pd.concat([df.drop(['verification_status', 'application_type',
                         'initial_list_status', 'purpose'], axis=1), dummies], axis=1)

print("Make some replacements in home ownership.")
df['home_ownership'] = df['home_ownership'].replace(["NONE", "ANY"], "OTHER")
print('\n')

dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis=1)

print("Column address is going to harm to our model, so let's drop it.")
print('\n')
df.drop('address', axis=1, inplace=True)

print("Do the same with the column issue.")
print('\n')
df.drop('issue_d', axis=1, inplace=True)

print("Change the columns earliest_cr_line as you want.")
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
print('\n')

print("We have ended our feature engineering!!!")
