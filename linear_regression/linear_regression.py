import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = datasets.make_regression(n_samples=200,
                                n_features=1,
                                noise=10,
                                random_state=2)
# グラフ表示
plt.plot(X, y, 'o')
plt.show()

# 訓練データとテストデータに分ける
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

# Linear_Regressionのモデル作成
lr_model = LinearRegression()

# 学習
lr_model.fit(X_train, y_train)

input("Press the <ENTER> key to continue and see the result...\n")

# 訓練データとテストデータの精度確認
lx = np.arange(-4, 4, 0.01)
ly = lr_model.coef_*lx + lr_model.intercept_
fig, ax = plt.subplots(2, 1)
ax[0].plot(lx, ly, 'r')
ax[0].plot(X_train, y_train, 'o')
ax[1].plot(lx, ly, 'r')
ax[1].plot(X_test, y_test, 'o')
plt.show()

print('訓練データの精度: ' + str(lr_model.score(X_train, y_train)))
print('テストデータの精度: ' + str(lr_model.score(X_test, y_test)))
print('回帰直線式: y = {} X + {}'.format(lr_model.coef_[0], lr_model.intercept_))

