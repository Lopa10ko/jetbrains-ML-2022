import pandas as pd

pd.set_option('display.max_columns', 8)
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
print(f"Data shape: {data.shape}")
# print(data.head(50))
print(data.sample(20, random_state=30))