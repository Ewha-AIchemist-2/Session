### ***2주차 실습***

주제 : 캐글에 대해 알아봅시다!   
이번 주차에서는 교재 실습에 자주 등장하는 캐글에 대한 전반적인 이해와 제출 방법을 알아보고    
직접 캐글 경진대회에 코드를 제출해보는 시간을 가지도록 하겠습니다!

***📔 실습 PPT***   
[AIchemist 2기 2주차_실습.pdf](https://github.com/Ewha-AIchemist-2/Session/files/14568562/AIchemist.2.2._.pdf)

📑 활용 자료   
< 캐글 Cereals >
[Kaggle_Cereals.zip](https://github.com/Ewha-AIchemist-2/Session/files/14568590/Kaggle_Cereals.zip)

< 캐글 Linear Regression Salary>
[Kaggle_Linear_Regression_Salary.zip](https://github.com/Ewha-AIchemist-2/Session/files/14568593/Kaggle_Linear_Regression_Salary.zip)

< 캐글 Titanic >
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

