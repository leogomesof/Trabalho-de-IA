import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carrega o dataset CIFAR-10
cifar10 = fetch_openml('cifar_10', version=1, cache=True)

# Separa os dados e os rótulos
X = cifar10.data.to_numpy()  # Converte para NumPy array
y = cifar10.target

# Converte os rótulos para inteiros
y = y.astype(np.uint8)

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza os dados (as imagens têm valores de pixel de 0 a 255, então dividimos por 255 para ter valores entre 0 e 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Inicializa o classificador de Regressão Logística
logistic_model = LogisticRegression(max_iter=100, solver='saga', multi_class='multinomial', n_jobs=-1)

# Treina o modelo
logistic_model.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = logistic_model.predict(X_test)

# Avalia o modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Exibe os resultados
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")