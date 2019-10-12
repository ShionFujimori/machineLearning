import numpy as np
import pandas as pd
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

print('\n説明変数名 (X): ' + str(iris.feature_names) + '\n')
print('目的変数名 (y): ' + str(iris.target_names) + '\n')

input("Press the <ENTER> key to continue and "
      "see the actual quantitative data...\n")
print('説明変数:\n' + str(iris.data) + '\n')
print('目的変数:\n' + str(iris.target) + '\n')

# 訓練データとテストデータに分ける
X_train, X_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

# kNNのモデル作成
knn = KNeighborsClassifier(n_neighbors=5)

# 学習
knn.fit(X_train, y_train)

# テストデータの予測
y_predict = knn.predict(X_test)

# 予測結果と正解の確認
input("Press the <ENTER> key to continue and see the result...\n")
print('=====予測結果と正解の確認=====\n')
print(str(pd.DataFrame({'予測': y_predict, '正解': y_test},
                       index=np.arange(1, len(y_predict)+1))) + '\n')
print('正解率: ' + str(np.mean(y_predict == y_test)*100) + '%')
