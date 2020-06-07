## Dacon FIFA Player Values prediction
- Dacon 해외 축구 선수 이적료 예측 미션
> 데이터 셋 기본 정보
 
- 부족한 점 review
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

# df.largest(n, columns, keep = "first")
# 지정한 열의 n개 만큼 내림 차순으로 값을 반환 한다.
# keep = "first" : 중복 값중 첫번째, "last" : 마지막 값, "all" : 모든 값 반환
# df.sort_values(columns, ascending = False).head(n)가 같지만 좀 더 능률이 좋다.

# df.reset_index
# df의 기존 index를 열에 추가 시키고, 새로운 순차 수열 index 생성
# df.reset_index(drop = True) : 기존 index 제거, 새로운 순차 수열 index 생성

# pd.merge()
# 두 객체에 중복된 컬럼 이름이 하나도 없다면 따로 지정해준다. 아래는 지정 방법이다.
# pd.merge(left_on)  : 조인키로 사용할 left DataFrame의 컬럼
# pd.merge(right_on) : 조인키로 사용할 right DataFrame의 컬럼
~~~
