## Dacon FIFA Player Values prediction
~~~python3
# np.inf = 무한대를 표시
pd.cut(df, [0, 1, np.inf], labels = ["a","b"])
# 0 <= a < 1, b >= 1
~~~
