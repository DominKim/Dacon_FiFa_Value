## Dacon FIFA Player Values prediction
> Dacon 해외 축구 선수 이적료 예측 미션
> - 데이터 셋 기본 정보
> ~~~ python3
> 종속변수(y) : value(연속형변수)
> 독립변수(x)
> - 연속형 변수
>   age, stat_overall, stat_potential
> - 비연속형 변수(범주형 변수)
>   contract_until, continent, position, prefer_foot, reputation, stat_skill_moves
> ~~~
> - EDA(Exploratory Data Analysis, 탐색적 데이터 분석)
> ~~~ python3
> - contract_until
> # 계약기간 변수는 숫자와 문자로 결합 되어 있는 변수이다. 이를 처리 하기위해 문자를 숫자 포함하고 있는 변수들로 아래와 같이 처리 하였다.
> comdata["contract_until"] = comdata["contract_until"].str.slice(-4,)
> comdata["contract_until"]
> # 위와 같이 변수를 전처리 후에 계약기간 변수의 범주 마다 빈도를 확인하면 2024년 이후의 범주는 2023년 이전 범주보다
> # 빈도가 꽤 낮은 것을 확인 할 수 있다. 이와 같은 이유때문에 2024년 이후 범주는 모두 2024년으로 아래와 같이 통일 시켰다.
> comdata.loc[comdata["contract_until"] == "2025", "contract_until"] = "2024"
> comdata.loc[comdata["contract_until"] == "2026", "contract_until"] = "2024"
>
> - continent
> # 대륙 변수의 범주별 빈도를 확인 하면 오세아니아가 상대적으로 빈도가 작은 걸 확인 할 수 있지만
> # 다른 빈도와 합치는 것은 모델링을 통해 RMSE점수를 확인 하고 시도해 볼 필요가 있다.
>
> - position
> # 포지션 변수는 당연하게도 골키퍼 포지션의 빈도수가 가장 작다.
> # 선수들은 포지션별로 가지는 특성이 다르기 때문에 본래대로 분류해서 사용할 필요가 있다.
>
> - prefer_foot
> # 왼발잡이가 상대적으로 오른발잡이 보다 빈도가 작다.
> # ttest를 통해 왼발과 오른발의 선수가치 평균을 확인 해보면 차이가 없는 것을 확인해 볼수있다.
> # 범주형 변수를 분석에 사용하기 위해서는 더미변수를 사용해 전처리가 필요한데, 이진속성 변수는 변수를 0,1로 변환해주면 된다.
> # replace 여러개의 변수를 처리 할 때는 [](리스트)를 사용하면 된다.
> comdata["prefer_foot"] = comdata["prefer_foot"].replace(["right", "left"], [0, 1])
>
> - reputation
> # 선수명성 변수는 숫자로 되어 있지만 1 ~ 5사이의 범주로 설정이 되어 있어 있는 범주형 변수이다.
> # 5점 범주가 많이 작은 빈도를 보인다. 이는 연속형 변수로 보면 이상치가 될 수도 있다. 실제 
> # 모델링을 통해 예측력을 높이기 위한 전처리 과정 중 하나가 될 수 있다.
>
> - stat_skill_moves
> # 개인기 변수는 선수명성 변수와 같은 숫자로 되어 있지만 1 ~ 5사이 5점척도로 되어 있는 범주형변수이다.
> # 또한, 5점 범주가 많이 작은 빈도를 보인다. 선수명성과 같이 전처리 과정 중 하나가 될 수 있다.
>
> - age
> # distplot으로 분포를 확인해보면 정규성을 보이는 것을 볼 수 있고 boxplot을 통해 이상치를
> # 확인해 보면 이상치가 존재 하는걸 볼 수 있다. 이상치를 해결 하기위한 전처리 과정이 필요하다.
>
> - stat_overall, stat_potential
> # 오버롤, 포텐션 변수를 displot으로 분포를 확인해 보면 평균보다 중앙값, 최빈수가 큰 정적편포를 보인다.
> # 또한, boxplot을 사용해 이상치를 확인해 보면 이상치가 존재 하는걸 불 수 있다. 이는 이상치를 해결 하기 위해
> # 전처리 과정이 필요하다는 것을 나타낸다.
> ~~~
> - 데이터 전처리
> ~~~ python3
> # age, stat_overall, stat_potential 변수는 RobustScaler를 사용해 표준화를 하였다.
> # RobustScaler : 특성들이 같은 스케일을 갖게 된다는 통계적 측면에서는 StandardScaler와 비슷하다.
> # 하지만 평균과 분산대신 중간값과 사분위값을 사용해 이상치에 영향을 받지 않는다.
>
> # One-Hot-encoding 더미변수 생성(범주형 변수)
> train_dummy = pd.get_dummies(train)
> real_test_dummy = pd.get_dummies(real_test)
> ~~~
> - model 평가
> ~~~ python3
> # Gridsearch로 확인 한 결과 xgboost 가장 좋은 예측력을 보였고 또한 특성공학을 통해
> # 최적의 파라미터를 찾았다.
> # rmse : 215126074267.982 정도의 점수가 나왔다. 좀 더 높은 점수를 얻기 위해 새로운 전처리 방법이 필요하다.
> # ex) 나이를 전성기에 접어든 나이와 그보다 어리고 많은 나이 그룹으로 나누기 등
> ~~~



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
