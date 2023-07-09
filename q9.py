import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import math
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# Carregar o conjunto de dados Iris
data = pd.read_csv('iris/iris.data', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Dividir o conjunto de dados em treinamento, teste e validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Usando heurísitca para chutar o número de neurônios na camada oculta
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
hidden_layer_sizes = math.ceil(math.sqrt(n_features + n_classes))
max_iter = 2500

# Definir o modelo MLP
mlp_no_hyper_adjust = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=42, max_iter=max_iter)

# Treinar o modelo MLP (sem ajuste)
start_time = time.time()
mlp_no_hyper_adjust.fit(X_train, y_train)
end_time = time.time()
processing_time_no_adjust = end_time - start_time

# Definir os possíveis valores dos hiperparâmetros a serem explorados
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive']
}

# Realizar a busca em grade dos melhores hiperparâmetros, usando validação cruzada com 10 folds
mlp = MLPClassifier(random_state=42, max_iter=2500)
grid_search = GridSearchCV(mlp, param_grid, cv=10, verbose=1)

start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()
processing_time_with_adjust = end_time - start_time

# Melhores hiperparâmetros
best_params = grid_search.best_params_
print("Melhores hiperparâmetros:", best_params)

# Treinar o modelo com os melhores hiperparâmetros, usando o conjunto de treinamento completo
mlp_best = MLPClassifier(**best_params, random_state=42)

start_time = time.time()
mlp_best.fit(X_train, y_train)
end_time = time.time()
processing_time_best = end_time - start_time

# Métricas de acurácia, precisão, recall e f1-score do modelo com os melhores hiperparâmetros e do modelo sem ajuste
print("Acurácia (sem ajuste):", mlp_no_hyper_adjust.score(X_test, y_test))
print("Acurácia (com ajuste):", mlp_best.score(X_test, y_test))

print("Precisão (sem ajuste):", precision_score(y_test, mlp_no_hyper_adjust.predict(X_test), average='macro'))
print("Precisão (com ajuste):", precision_score(y_test, mlp_best.predict(X_test), average='macro'))

print("Recall (sem ajuste):", recall_score(y_test, mlp_no_hyper_adjust.predict(X_test), average='macro'))
print("Recall (com ajuste):", recall_score(y_test, mlp_best.predict(X_test), average='macro'))

print("F1-score (sem ajuste):", f1_score(y_test, mlp_no_hyper_adjust.predict(X_test), average='macro'))
print("F1-score (com ajuste):", f1_score(y_test, mlp_best.predict(X_test), average='macro'))

# Print the processing time for each model
print("Tempo de processamento (sem ajuste):", processing_time_no_adjust)
print("Tempo de processamento (com ajuste):", processing_time_best)


# Resultados:
# Acurácia (sem ajuste): 1.0
# Acurácia (com ajuste): 0.9
# Precisão (sem ajuste): 1.0
# Precisão (com ajuste): 0.923076923076923
# Recall (sem ajuste): 1.0
# Recall (com ajuste): 0.9
# F1-score (sem ajuste): 1.0
# F1-score (com ajuste): 0.8976982097186701
# Tempo de processamento (sem ajuste): 0.3797750473022461
# Tempo de processamento (com ajuste): 0.04429793357849121

# Links da 6 e 8
# 6 https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10015228
# 8 https://repositorio.ufu.br/bitstream/123456789/37777/3/Implementa%c3%a7%c3%a3odeAutocomplementos.pdf