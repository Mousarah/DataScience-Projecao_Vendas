# Importações
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def createboxplot(dataframe, eixox):
    sns.boxplot(x=eixox, y='vendas', data=dataframe)
    plt.show()


# Importação e limpeza dos dados de vendas
df = pd.read_csv('vendas.csv', delimiter=",")
df.dropna(inplace=True)

# Análise exploratória dos dados
describe = df.describe().to_string()

# Análise de tendências nas vendas ao longo do tempo
df = df.sort_values(by='data')
df['data'] = pd.to_datetime(df['data'], format='%Y/%m/%d').dt.strftime('%d/%m/%Y')
df.set_index('data', inplace=True)
plt.title("Tendência de vendas ao longo do tempo")
plt.text(df.index[0], df['vendas'][0], describe, bbox=dict(facecolor='white', alpha=0.8))
sns.set_style("whitegrid")
sns.lineplot(x='data', y='vendas', data=df, linewidth=2.5)
plt.xticks(range(0, len(df), 3), df.index[::3])
plt.show()

# Segmentação dos clientes e análise do comportamento de compra
createboxplot(df, 'regiao')
createboxplot(df, 'idade')
createboxplot(df, 'genero')

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
df_result = pd.DataFrame({"Data": pd.array(y_test.index), "Vendas": y_pred})
df_result = df_result.sort_values(by='Data')

mse = 'Erro médio quadrático: %.2f' % mean_squared_error(y_test, y_pred)
cod = 'Coeficiente de determinação: %.2f' % r2_score(y_test, y_pred)

plt.title("Previsão de vendas")
plt.text(df_result['Data'][0], df_result['Vendas'][0], mse + '\n' + cod, bbox=dict(facecolor='white', alpha=0.8))
sns.set_style("whitegrid")
sns.lineplot(x='Data', y='Vendas', data=df_result, linewidth=2.5)
plt.show()
