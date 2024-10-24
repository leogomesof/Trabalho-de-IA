import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
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

# Inicializa o classificador de Random Forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

# Treina o modelo
random_forest.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = random_forest.predict(X_test)

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