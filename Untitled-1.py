# %%
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# %%
dataset = pd.read_csv('data/healthcare-dataset-stroke-data.csv' , sep=',')

dataset = dataset.drop(dataset.columns[0], axis=1)

dataset = dataset.dropna()

dataset.info()


# %%
dataset["gender"] = dataset["gender"].replace(["Other"], "Female")
gender_conversion = {"Male": 0, "Female": 1}
dataset["gender"] = dataset["gender"].map(gender_conversion)
dataset["gender"] = dataset["gender"].astype(int)


# %%
married_conversion = {"No": 0, "Yes": 1}
dataset["ever_married"] = dataset["ever_married"].map(married_conversion)
dataset["ever_married"] = dataset["ever_married"].astype(int)


# %%
df_work_ohe = pd.get_dummies(
    dataset["work_type"], 
    prefix="work_ohe", 
    drop_first=True,
)
dataset = pd.concat([dataset, df_work_ohe], axis=1)
dataset = dataset.drop(["work_type"], axis=1)


# %%
residence_conversion = {"Rural": 0, "Urban": 1}
dataset["Residence_type"] = dataset["Residence_type"].map(residence_conversion)
dataset["Residence_type"] = dataset["Residence_type"].astype(int)


# %%
df_smoking_ohe = pd.get_dummies(
    dataset["smoking_status"], 
    prefix="smoking_ohe", 
    drop_first=True,
)
df_smoking_ohe = df_smoking_ohe.rename(columns={
    "smoking_ohe_never smoked": "smoking_ohe_never_smoked", 
    "smoking_ohe_formerly smoked": "smoking_ohe_formerly_smoked",
})
dataset = pd.concat([dataset, df_smoking_ohe], axis=1)
dataset = dataset.drop(["smoking_status"], axis=1)


# %%
x = dataset.iloc[:, 0:10].values
y = dataset.iloc[:, 10].values


# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# %%
model = Sequential()
model.add(Dense(6, input_dim=10, activation='relu', kernel_initializer='uniform'))
model.add(Dense(4, activation='relu', kernel_initializer='uniform'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))


# %%
model.summary()


# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# %%
model.fit(x_train, y_train, epochs=150, batch_size=10)


# %%
loss, accuracy = model.evaluate(x_test, y_test)
print("\nLoss: %.2f, Acurácia: %.2f%%" % (loss, accuracy*100))

# %%
predictions = model.predict(x)

# %%
# Ajusta as previsões e imprime o resultado
previsões = [round(x[0]) for x in predictions]
print(previsões)


