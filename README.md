## Dacon FIFA Player Values prediction
~~~python3
# np.inf = 무한대를 표시
pd.cut(df, [0, 1, np.inf], labels = ["a","b"])
# 0 <= a < 1, b >= 1

# count() vs size() vs nunique()
# count() : 결측치를 포함하지 않은 값 반환
# size() : 결측치를 포함한 값의 길이 반환
# nunique() : 결측치를 제외한 유일값 리턴

# df.rename(columns = {"기존 열 이름" : "새로운 열 이름"})
# df의 열이름을 변경할때 dict형을 이용하면 쉽게 바꿀수 있이다.
~~~
