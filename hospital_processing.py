import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 20)
general_data = pd.read_csv("./test/general.csv")
prenatal_data = pd.read_csv("./test/prenatal.csv")
sports_data = pd.read_csv("./test/sports.csv")
prenatal_data = prenatal_data.rename(columns={'HOSPITAL': 'hospital', 'Sex': 'gender'})
sports_data = sports_data.rename(columns={'Hospital': 'hospital', 'Male/female': 'gender'})

data = pd.concat([general_data, prenatal_data, sports_data], ignore_index=True).drop(columns=['Unnamed: 0'])
data.dropna(axis=0, how='all', inplace=True)
data['gender'].replace({'female': 'f', 'male': 'm',
                        'woman': 'f', 'man': 'm'}, inplace=True)
# data.loc[data['hospital'] == 'prenatal', 'gender'].fillna('f', inplace=True)
data.loc[data['hospital'] == 'prenatal', 'gender'] = data.gender.loc[data['hospital'] == 'prenatal'].fillna('f')
data['bmi'].fillna(0, inplace=True)
data['diagnosis'].fillna(0, inplace=True)
data['blood_test'].fillna(0, inplace=True)
data['ecg'].fillna(0, inplace=True)
data['ultrasound'].fillna(0, inplace=True)
data['mri'].fillna(0, inplace=True)
data['xray'].fillna(0, inplace=True)
data['children'].fillna(0, inplace=True)
data['months'].fillna(0, inplace=True)
# print(f"Data shape: {data.shape}")
# print(data.sample(20, random_state=30))
# print(data.head())

data_general = data.loc[data.hospital == 'general', ::]
data_prenatal = data.loc[data.hospital == 'prenatal', ::]
data_sports = data.loc[data.hospital == 'sports', ::]

'''1.1 Which hospital has the highest number of patients?'''
ans_1 = max(hospital_qset := dict(data.groupby(['hospital']).gender.count()), key=hospital_qset.get)

'''1.2 What share of the patients in the general hospital suffers from stomach-related issues?'''
ans_2 = round(data_general.loc[data_general.diagnosis == 'stomach', ::].shape[0] / len(data_general), 3)

'''1.3 What share of the patients in the sports hospital suffers from dislocation-related issues?'''
ans_3 = round(dict(data_sports.diagnosis.value_counts())['dislocation'] / len(data_sports), 3)

'''1.4 What is the difference in the median ages of the patients in the general and sports hospitals?'''
mean_age_hospitalwise = dict(data.pivot_table(data, index='hospital')['age'])
ans_4 = mean_age_hospitalwise['general'] - mean_age_hospitalwise['sports']

'''2.1 What is the most common age among all hospitals?'''
data['age'].plot(kind='hist', bins=12)
plt.show()

'''2.2 What is the most common diagnosis among patients in all hospitals?'''
# data['diagnosis'].value_counts().plot(kind='bar')
plt.pie(x=list(data['diagnosis'].value_counts()), labels=data['diagnosis'].unique())
plt.show()

'''2.3 Build a violin plot of height distribution by hospitals'''
fig, axes = plt.subplots()
violin = plt.violinplot(data['height'].values, showmeans=True, showextrema=True, showmedians=True)
axes.set_title('Height by hospitals')
plt.show()
print(f"The answer to the 1st question: {ans_1}\n"
      f"The answer to the 2st question: {ans_2}\n"
      f"The answer to the 3rd question: {ans_3}\n"
      f"The answer to the 4th question: {ans_4}")
