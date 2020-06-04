## Dacon FIFA Player Values prediction
~~~python3
# np.inf = 무한대를 표시
pd.cut(df, [0, 1, np.inf], labels = ["a","b"])
# 0 <= a < 1, b >= 1

# count() vs size() vs nunique()
# count() : 결측치를 포함하지 않은 값 반환
# size() : 결측치를 포함한 값의 길이 반환
# nunique() : 결측치를 제외한 유일값 리턴
~~~
