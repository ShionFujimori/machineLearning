import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

X, y = datasets.make_regression(n_samples=100,
                                n_features=200,
                                noise=20,
                                random_state=0)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=1)

# 線形回帰 (正則化なし)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print('\n正則化なしの場合:\n')
print('訓練データの精度: ' + str(lr_model.score(X_train, y_train)))
print('テストデータの精度: ' + str(lr_model.score(X_test, y_test)))

input("\nPress the <ENTER> key to continue...\n")

# Ridge回帰
# 係数のL2ノルム(二乗の和)に対してペナルティを与える
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
print('Ridge回帰の場合:\n')
print('訓練データの精度: ' + str(ridge_model.score(X_train, y_train)))
print('テストデータの精度: ' + str(ridge_model.score(X_test, y_test)))

input("\nPress the <ENTER> key to continue...\n")

# Lasso回帰
# 係数のL1ノルム(絶対値の和)に対してペナルティを与える
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
print('Lasso回帰の場合:\n')
print('訓練データの精度: ' + str(lasso_model.score(X_train, y_train)))
print('テストデータの精度: ' + str(lasso_model.score(X_test, y_test)))

