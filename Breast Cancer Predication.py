#Breast Cancer Predication

#Bibliotecas de Machine Learning
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Bibliotecas de upload, leitura e tratamento de dados
import matplotlib.pyplot as plt
from google.colab import files
import io
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------------------------------------------

#Fazer upload do banco de dados
data = files.upload()

#-----------------------------------------------------------------------------------------------------------

#Leitura do arquivo Breast Cancer Prediction.csv
project_data = pd.read_csv(io.BytesIO(data['Breast Cancer Prediction.csv']),sep= ';')
project_data = project_data.drop(columns=["id"]) #Remoção de dados de cadastro dos pacientes
project_data.head() #Plot do Cabeçalho

#-----------------------------------------------------------------------------------------------------------

project_data.rename(columns = {"diagnosis":"Diagnóstico","Radius_mean":"Raio Médio","Texture_mean":"Textura Média","perimeter_mean":"Perímetro Médio","area_mean":"Área Média","smoothness_mean":"Suavidade Média",
                               "compactness_mean":"Compacidade Média","concavity_mean":"Concavidade Média","concave points_mean":"Pontos Côncavos Médios","symmetry_mean":"Simetria Média",
                               "fractal_dimension_mean":"Média da Dimensão Fractal",
                               "radius_se":"Erro Padrão do Raio","texture_se":"Erro Padrão da Textura","perimeter_se":"Erro Padrão do Perímetro","area_se":"Erro Padrão da Área","smoothness_se":"Erro Padrão da Suavidade",
                               "compactness_se":"Erro Padrão da Compacidade","Concavity_se":"Erro Padrão da Concavidade","concave points_se":"Erro Padrão dos Pontos Côncavos","symmetry_se":"Erro Padrão da Simetria",
                               "fractal_dimension_se":"Erro Padrão da Dimensão Fractal",
                               "radius_worst":"Pior Raio","texture_worst":"Pior Textura","perimeter_worst":"Pior Perímetro","area_worst":"Pior Área","smoothness_worst":"Pior Suavidade",
                               "compactness_worst":"Pior Compacidade","concavity_worst":"Pior Concavidade","concave points_worst":"Piores Pontos Côncavos","symmetry_worst":"Pior Simetria",
                               "fractal_dimension_worst":"Pior Dimensão Fractal"}, inplace = True)
project_data.head() #Plot do Cabeçalho

#-----------------------------------------------------------------------------------------------------------

#Histograma da quantidade de amostras de Nódulos Benígnos e Malígnos
print("Quantidade de amostras:",len(project_data["Diagnóstico"]))
print("Distribuição dos casos Benígnos(B) e Malígnos(M):")
plt.hist(project_data["Diagnóstico"], color = "g")
plt.show()

#-----------------------------------------------------------------------------------------------------------

#Manipulação dos dados
n_columns = project_data.columns[1:]
print(n_columns)

#-----------------------------------------------------------------------------------------------------------

#Identificando os dados correspondentes de cada classe
p_dM = project_data.loc[project_data['Diagnóstico'] == "M"]
p_dB = project_data.loc[project_data['Diagnóstico'] == "B"]
dM = p_dM.values
dB = p_dB.values

#-----------------------------------------------------------------------------------------------------------

#Gerar Gráficos de Densidade de Dados
for i in range(1,len(project_data.columns)):
  plt.style.use('bmh')
  plt.hist(dM[:,i],color = 'r', alpha=0.6)
  plt.hist(dB[:,i],color = 'b', alpha=0.6)
  plt.legend(["Sine","Cosine"])
  plt.title(project_data.columns[i])
  plt.legend(["M","B"])
  plt.show()

#-----------------------------------------------------------------------------------------------------------

#Gerar Gráficos de Densidade de Dados
Y = project_data["Diagnóstico"]
print(Y)
for i in range(0,len(Y)):
  if Y[i] == "M":
    Y[i] = -1
  if Y[i] == "B":
    Y[i] = 1

print(Y)

#-----------------------------------------------------------------------------------------------------------

#Treinamento e o Teste para a regressão linear

prop_test = 0.25
X = project_data.drop(columns=["Diagnóstico"])
Y = project_data["Diagnóstico"]
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size = prop_test)

model_reg_linear = LinearRegression()
model_reg_linear.fit(X_train,Y_train)

#-----------------------------------------------------------------------------------------------------------

#Acurácia do treinamento
Y_hat_train = model_reg_linear.predict(X_train)
D_hat_train = np.sign(Y_hat_train)
error_train = np.abs((D_hat_train - Y_train)/2)
acc_train = 1 - (np.sum(error_train)/error_train.size)
print("Acuracia de treinamento:", acc_train)

#Acurácia do teste
Y_hat_test = model_reg_linear.predict(X_test)
D_hat_test = np.sign(Y_hat_test)
error_test = np.abs((D_hat_test - Y_test)/2)
acc_test = 1 - (np.sum(error_test)/error_test.size)
print("Acuracia de teste:", acc_test)

#-----------------------------------------------------------------------------------------------------------

#Coeficientes da regressão linear
print(model_reg_linear.coef_)

#-----------------------------------------------------------------------------------------------------------

#Rearranjo Valores para gerar a Matrix de Confusão

Y_train = np.array(Y_train)

for i in range(0,len(D_hat_train)):
  if D_hat_train[i] == -1:
    D_hat_train[i] = 0
  if Y_train[i] == -1:
    Y_train[i] = 0
 	
#-----------------------------------------------------------------------------------------------------------
 
#Matrix de Confusão
 
Y_train = np.float64(Y_train)
Conf_Matriz = confusion_matrix(Y_train,D_hat_train)
disp = ConfusionMatrixDisplay(confusion_matrix = Conf_Matriz)
disp.plot()
plt.show()
 
#-----------------------------------------------------------------------------------------------------------
 
#Curva ROC treinamento
fpr,tpr,thresholds = roc_curve(Y_train,Y_hat_train)
plt.plot(fpr,tpr)
auc = roc_auc_score(Y_train, Y_hat_train)
print('AUC: %.3f' % auc)