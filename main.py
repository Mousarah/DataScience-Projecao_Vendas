# Importações
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# Importação e limpeza dos dados de vendas
df = pd.read_csv('vendas.csv')
df.dropna(inplace=True)

# Análise exploratória dos dados
print(df.describe())
sns.pairplot(df)
plt.show()

# Análise de tendências nas vendas ao longo do tempo
df['data'] = pd.to_datetime(df['data'])
df.set_index('data', inplace=True)
df['vendas'].plot()
plt.show()

# Segmentação dos clientes e análise do comportamento de compra
sns.boxplot(x='regiao', y='vendas', data=df)
plt.show()
sns.boxplot(x='idade', y='vendas', data=df)
plt.show()
sns.boxplot(x='genero', y='vendas', data=df)
plt.show()

# Transformação das colunas categóricas em variáveis dummy
df = pd.get_dummies(df, columns=['regiao', 'idade', 'genero'])

# Preparação dos dados para previsão das vendas futuras
X = df.drop(['vendas'], axis=1)
y = df['vendas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Carrega o modelo treinado
if os.path.isfile('modelo.pkl'):
    model = joblib.load('modelo.pkl')
else:
    model = LinearRegression()

# Treina o modelo
model.fit(X_train, y_train)

# Salva o modelo treinado
joblib.dump(model, 'modelo.pkl')

# Previsão das vendas futuras
y_pred = model.predict(X_test)
print('Erro médio quadrático: %.2f' % mean_squared_error(y_test, y_pred))
print('Coeficiente de determinação: %.2f' % r2_score(y_test, y_pred))
