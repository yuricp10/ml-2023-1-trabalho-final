import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from keras.models import Sequential 
from keras.layers import Dense,Dropout 
from keras.utils import np_utils



#Carregando o dataset
data=pd.read_csv("iris.csv")
print("Descrevendo os dados: ",data.describe())
print("10 primeiros exemplos do dataset:",data.head(10))
print("10 últimos exemplos do dataset:",data.tail(10))


#Fazendo cada espécie corresponder a um valor numérico
data.loc[data["Species"]=="Iris-setosa","Species"]=0
data.loc[data["Species"]=="Iris-versicolor","Species"]=1
data.loc[data["Species"]=="Iris-virginica","Species"]=2



#Armazenando o número de classes para servir como base para ser o número de neuronios na camada de saída
numero_classes=len(data["Species"].unique())

print(data.head())


#Convertendo "data" para arrays numpy para que possa ser processado 
# X é o array dos dados de entrada e y é o array dos dados de saída

#armazenando o número de dados de entrada para ser definido como o número de neuronios da camada de entrada
numero_dados_entrada=len(data.columns)-1

X=data.iloc[:,0:numero_dados_entrada].values
y=data.iloc[:,numero_dados_entrada].values

print("Formato de X",X.shape)
print("Formato de y",y.shape)
print("Exemplos de X\n",X[:3])
print("Exemplos de y\n",y[:3])


#Normalizando os dados
transformer = Normalizer().fit(X)
X_normalized=transformer.transform(X)

print("Exemplos de x normalizado:\n",X_normalized[:3])

#divide o dataset em treinamento e teste de forma mais simples que o código acima

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2)

print("Comprimento do conjunto de treinamento de x:",X_train.shape[0],"e y:",y_train.shape[0])
print("Comprimento do conjunto de teste de x:",X_test.shape[0],"e y:",y_test.shape[0])

#Convertendo y_train e y_test em codificação one-hot
y_train=np_utils.to_categorical(y_train,num_classes=numero_classes)
y_test=np_utils.to_categorical(y_test,num_classes=numero_classes)
print("Formato de y_train:",y_train.shape)
print("Formato de y_test:",y_test.shape)


#Criando a rede neural
model=Sequential()

model.add(Dense(50,input_dim=numero_dados_entrada,activation='relu'))

model.add(Dense(30,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(15,activation='relu'))

model.add(Dense(numero_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Apresenta o resumo da rede
model.summary()


#preparando para realizar o early stopping e salvar a rede neural que obteve o melhor resultado durante o treinamento
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

numero_de_epocas=50

#earling stop
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001,restore_best_weights=True)

#salvando o melhor modelo 
mc = ModelCheckpoint('melhor_modelo.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


history = model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=numero_de_epocas,verbose=1,callbacks=[es,mc])



#verificando desempenho da rede neural nos dados de teste

#Realizando a estimação sobre os dados de teste
prediction=model.predict(X_test)
#Cálculo da acurácia/precisao (Accuracy) do teste
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)
precisao=np.sum(y_label==predict_label)/length * 100


print("Resultado precisao:",precisao )
print(history.history.keys())

from keras.models import load_model
# Carrega o modelo e arquitetura
model = load_model("melhor_modelo.h5")

print("Modelo carregado do disco")
# summarize model.
model.summary()

#Estimação dos valores
prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)

precisao=np.sum(y_label==predict_label)/length * 100 

print("Relatório de Classificação nos Dados de Teste")
print(classification_report(predict_label, y_label))
print("precisao dos testes feitos na rede neural carregada:",precisao )
print("Predição obtida pela rede:",predict_label)
print("Predição esperada:        ",y_label)



#Guarda as classes no formato numérico (0,1,2,...)
classes_numericas=data['Species'].unique()

#Testa com a matriz de confusão
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

matriz_confusao = confusion_matrix(predict_label, y_label,labels=classes_numericas)

print("Matriz de Confusão do Teste Realizado")
print(matriz_confusao)





    