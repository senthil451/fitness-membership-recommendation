import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("gym prediction.csv")
af=pd.read_excel('additional.xlsx')
x = df[['Age','Gender','Goal_Type','Weekly_Visits',
        'Diet_Plan','Personal_Training','Previous_Surgery']]
y_price = df['Price']
y_plan = df['Recommended_Plan']

# Price Prediction Model

lr_model = LinearRegression()
lr_model.fit(x, y_price)

age = int(input('Enter your Age: '))
gender = int(input('Enter your gender (1=Male, 0=Female): '))
goal = int(input('Enter your goal (1=Gain, 0=Loss): '))
visit = int(input('Weekly visits (days): '))
diet = int(input('Diet Plan (1/0): '))
pt = int(input('Personal Training (1/0): '))
ps = int(input('Previous Surgery (1/0): '))

new_input = pd.DataFrame([[age, gender, goal, visit, diet, pt, ps]],
    columns=['Age','Gender','Goal_Type','Weekly_Visits','Diet_Plan','Personal_Training','Previous_Surgery'])

predicted_price = int(lr_model.predict(new_input)[0])
print("Predicted Price:", predicted_price)

# Plan Recommendation Model

X_train, X_test, y_train, y_test = train_test_split(x, y_plan,test_size=0.2,random_state=42)

rf_model = RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')

rf_model.fit(X_train, y_train)

plan = rf_model.predict(new_input)[0]
print("Recommended Plan:", plan)

s=''
if predicted_price >=500 and predicted_price<7000:
    s+='Basic'
elif predicted_price<14000:
    s+='Medium'
else:
    s+='Premium'
#Age,Gender,Goal_Type,Weekly_Visits,Diet_Plan,Personal_Training,Previous_Surgery,Price,Recommended_Plan,Category
new_row = pd.DataFrame([[age, gender, goal, visit, diet, pt, ps,predicted_price, plan, s]],
                       columns=['Age','Gender','Goal_Type','Weekly_Visits','Diet_Plan','Personal_Training','Previous_Surgery','Price','Recommended_Plan','Category'])

af= pd.concat([af,new_row],ignore_index=True)
df=pd.concat([df,new_row],ignore_index=True)

df.to_csv('gym prediction.csv',index=False)
af.to_excel('additional.xlsx',index=False)


plt.figure(figsize=(15,7))

plt.subplot(1,3,1)
gb=df.groupby('Category')['Gender'].count()
wedges, texts, autotexts = plt.pie(gb,labels=gb.index,autopct='%1.1f%%',wedgeprops={'edgecolor': 'white', 'linewidth': 2})
plt.title('Count of people by Category',fontweight='bold',color='red',fontstyle='oblique')
plt.legend(gb.index,loc='upper left',fontsize=8)
texts[0].set_color('red')      # Basic
texts[1].set_color('blue')     # Medium
texts[2].set_color('green')    # Premium

plt.subplot(1,3,2)
p=df.groupby('Recommended_Plan')['Price'].count()
plt.pie(p,labels=['1 month','3 month','6 month','12 month'],autopct='%1.1f%%',wedgeprops={'width': 0.4})
plt.title('Personal Training wise Price')

plt.subplot(1,3,3)
sns.boxplot(x='Category', y='Price', data=df)

plt.subplot(1,2,1)
df.groupby('Category')['Price'].mean().plot(kind='bar')
plt.ylabel("Average Price")
plt.title("Average Price by Category")

plt.suptitle("Gym Membership Analysis Dashboard",color='red',fontweight='bold',fontstyle='oblique',fontsize=15)
plt.tight_layout()
plt.show()
