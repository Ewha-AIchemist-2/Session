# 💠 AIchemist 11주차 

## 🌻 정모 안내
**1. 장소 : 아산공학관 152호**   
**2. 시간 : (월) 19:00 ~ 21:00**

## 코드 수정
p.605 교재 코드대로 하면 오류 발생합니다.
```python
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환. 
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x : (' ').join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.shape)
```
<br>

`count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))`
에서 에러 발생. 아래 코드로 수정 <br>
`count_vect = CountVectorizer(min_df=1, ngram_range=(1,2))`

- 수정 코드 
```python
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환. 
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x : (' ').join(x))
count_vect = CountVectorizer(min_df=1, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.shape)
```


## 🖋 과제
1. 파이썬 머신러닝 완벽가이드 : 텍스트 분석 - p.584 ~ 646 이론 열심히 공부해오기
2. 이론 PPT 빈칸 채운 뒤 PDF 파일로 깃헙에 제출
3. p.601 ~ 612 **콘텐츠 기반 필터링 실습 - TMDB 5000 영화 데이터 세트** 필사 후 .py 파일로 깃헙에 제출

<br>

**이론 PPT**<br>
[알켐_LN11.pdf](https://github.com/user-attachments/files/15857971/_LN11.pdf)


**참고 자료**
1. TMDB 5000 <br>
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

3. MovieLens Dataset <br>
https://grouplens.org/datasets/movielens/latest/


## 🚨 알림

## 🌱 Commit 규칙   
### Commit Convention      
    [Week n] 단원 이름 - PPT 빈칸 채우기   
    [Week n] 단원 이름 - {교재 예제 이름} 코드 필사      
+ 해당 주차 branch 생성 및 전환하기 
+ 본인 Repository에 n주차 폴더 생성 후 해당 폴더 내에 과제 Commit하기   
## 🌱 PR 규칙          
### PR Convention
    <제목>
    [Week n] 단원 이름 - 이론 세션
    <내용>
    ### 이름   
    이름   
    ### 과제   
    PPT 빈칸 채우기 : O / X
    교재 실습 예제 필사 : O / X
+ 본인의 repository의 해당 주차 branch에서 main branch로 PR을 날려준 뒤 PR 링크를 보내주세요
+ 운영진의 과제 확인 후 merge를 진행해주세요 

## 🗄 Repository 구조
```bash
📁 n주차
ㄴ📁 이론
  ㄴ📋 과제 파일
ㄴ📁 실습
  ㄴ📋 실습 파일
```

## 🚨 주의사항   
1. 모든 Commit은 항상 해당 주차의 branch로 전환 후에 진행해주세요!
2. Convention을 잘 지켜서 commit 및 PR을 진행해주세요!

