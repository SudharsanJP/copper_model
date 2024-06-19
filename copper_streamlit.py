import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
st.title(':orange[copper model datascience project]')

#) reading the dataset
df = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\copper_dataset\copper_dataset.csv")

#) dropping the unnecessary columns
df.drop(['id','material_ref','product_ref'],axis=1,inplace=True)

#) to check the null values
df.isna().sum()

#) forward fill and backward fill
temp_df = df.ffill()
temp_df = df.bfill()

#) to check the null values after fill
temp_df.isna().sum()

#) converting item date into item  year
list_item_date = temp_df['item_date'].tolist() #) converting item_date column into list
item_string = map(str, list_item_date)
list_item_string = list(item_string)

index = 0
list_item_year = []

while index < len(list_item_string):
    year1 = list_item_string[index]
    #) to pick up year
    item_year = year1[0:4]
    list_item_year.append(item_year)
    index+=1

#) converting list into column
temp_df['item_year'] = list_item_year

#) to check the type of data in each column
datatypes = df.dtypes

#)converting str column to int column 
temp_df['item_year'] = temp_df['item_year'].astype(str).astype(int)

#) converting item date into item month
list_item_date = temp_df['item_date'].tolist()
item_string = map(str, list_item_date)
list_item_string = list(item_string)

index = 0
list_item_month = []

while index < len(list_item_string):
    month1 = list_item_string[index]
    #) to pickup the month
    item_month = month1[4:6]
    list_item_month.append(item_month)
    index+=1

temp_df['item_month'] = list_item_month

#)converting str to int
temp_df['item_month'] = temp_df['item_month'].astype(str).astype(int)

#) converting item date into item  year
list_dvery_date = temp_df['delivery date'].tolist() 
dvery_string = map(str, list_dvery_date)
list_dvery_string = list(dvery_string)

index = 0
list_dvery_year = []

while index < len(list_dvery_string):
    year2 = list_dvery_string[index]
    #) to pick up year
    dvery_year = year2[0:4]
    list_dvery_year.append(dvery_year)
    index+=1

#) converting list into column
temp_df['dvery_year'] = list_dvery_year

#)converting str column to int column 
temp_df['dvery_year'] = temp_df['dvery_year'].astype(str).astype(int)

#) converting item date into item month
list_dvery_date = temp_df['delivery date'].tolist() 
dvery_string = map(str, list_dvery_date)
list_dvery_string = list(dvery_string)

index = 0
list_dvery_month = []

while index < len(list_dvery_string):
    month1 = list_dvery_string[index]
    #) to pickup the month
    dvery_month = month1[4:6]
    list_dvery_month.append(dvery_month)
    index+=1

temp_df['dvery_month'] = list_dvery_month

#)converting str to int
temp_df['dvery_month'] = temp_df['dvery_month'].astype(str).astype(int)

#) dropping the unnecessary columns
temp_df.drop(['item_date','delivery date'],axis=1,inplace=True)

#) to show the data frame
st.markdown("\n#### :blue[1.1 copper model dataset:]\n")
st.dataframe(temp_df.head(6))

#) to know the number of data
st.markdown("\n#### :blue[1.2 number of data points:]\n")
st.code(len(temp_df))

#) a string is available in quantity columns at row 73086
#temp_df.loc[173086]

#) to check the number of string data in quantity tons column
#temp_df[temp_df['quantity tons'] == 'e']
#) 28. button with text

#) drop string data in quantity tons column
temp_df.drop(173086,axis=0,inplace=True)

st.markdown("\n#### :blue[2.1 Iteration 1:]\n")
if (st.button(':red[click here]')):
    fig = px.box(temp_df,y="selling_price")
    st.plotly_chart(fig)

#) spotting the outliers in selling price column - max
#temp_df['selling_price'].max()

#) mean of selling price column
#temp_df['selling_price'].mean()

#) spotting the outliers in selling price column - min
#temp_df['selling_price'].min()

#) checking selling price beyond 10000
#temp_df[temp_df['selling_price']> 10000]

#) dropping the column
temp_df.drop([10228,36192,123570,124547],axis=0,inplace=True)

#) to check selling price less than 0
#temp_df[temp_df['selling_price']< 0]

#) to drop the row data of selling price less than 0
temp_df.drop([28,44761,44810,44865,105189],axis=0,inplace=True)

st.markdown("\n#### :blue[2.2 Iteration 2:]\n")
selectBox=st.selectbox("iteration: ", ['first','second'])
if selectBox == 'first':
    st.write("selctect option second")
elif selectBox == 'second':
    fig = px.box(temp_df,y="selling_price")
    st.plotly_chart(fig)

#) counting the row data of selling price beyond 4000
#temp_df[temp_df['selling_price']> 4000]

#) dropping down the row data of selling price beyond 4000
temp_df.drop([76871,113918,116445,141122,153215,159869],axis=0,inplace=True)

#) 21. checkbox with text
st.markdown("\n#### :blue[2.3 Iteration 3:]\n")
if (st.checkbox("iteration 3")):
    fig = px.box(temp_df,y="selling_price")
    st.plotly_chart(fig)

#) to check dvery year == 3031
#temp_df[temp_df['dvery_year']== 3031]

#) drop the column of year == 3031
temp_df.drop(58,axis=0,inplace=True)

#) 30. text input
st.markdown("\n#### :blue[2.4 Iteration 4:]\n")
textInput = st.text_input("Enter four:")
if (textInput):
    fig = px.box(temp_df,x="dvery_year",y="selling_price")
    st.plotly_chart(fig)

#) to know the dtype of each column
#temp_df.info()

#)converting str to int
temp_df['quantity tons'] = temp_df['quantity tons'].astype(str).astype(float)

#) after converting a column to float, to know the dtype of each column
#temp_df.info()

#) 31. password input
st.markdown("\n#### :blue[2.5 Iteration 5:]\n")
passInput = st.text_input("password:", type = 'password')
if (passInput):
    fig = px.box(temp_df,x="dvery_year",y="quantity tons")
    st.plotly_chart(fig)

#) max value in quantity tons column
#temp_df['quantity tons'].max()

#) min value in quantity tons column
#temp_df['quantity tons'].min()

#) mean value in quantity tons column
#temp_df['quantity tons'].mean()

#) to get the data of quantity tons beyond 100000
#temp_df[temp_df['quantity tons']> 100000]

#) drop the data of quantity tons beyond 100000
temp_df.drop([173022,173211],axis=0,inplace=True)

#) 32. input number
st.markdown("\n#### :blue[2.6 Iteration 6:]\n")
num_input = st.number_input("Enter the iteration number")
if (num_input):
    fig = px.box(temp_df,x="dvery_year",y="quantity tons")
    st.plotly_chart(fig)

#) to get the data of quantity tons beyond 30000
#temp_df[temp_df['quantity tons']> 30000]

#) drop the data of quantity tons beyond 30000
temp_df.drop([100256,100260,175802,176479],axis=0,inplace=True)


#) 29. Slider
st.markdown("\n#### :blue[2.7 Iteration 7:]\n")
vol = st.slider("select the range", 6,7)
if vol == 7:
    fig = px.box(temp_df,x="dvery_year",y="quantity tons")
    st.plotly_chart(fig)

#) to get the data of quantity tons beyond 2400
#temp_df[temp_df['quantity tons']> 24000]

#) drop the data of quantity tons beyond 24000
temp_df.drop([71486,100248],axis=0,inplace=True)

#) 36. time 
st.markdown("\n#### :blue[2.8 Iteration 8:]\n")
time = st.time_input("time")
if (time):
    fig = px.box(temp_df,x="dvery_year",y="quantity tons")
    st.plotly_chart(fig)


st.sidebar.header(":red[Iterations 9 - 17]")
if st.sidebar.button(":green[Show]"):
    #) boxplot
    
    st.markdown("\n#### :blue[2.9 Iteration 9:]\n")
    fig = px.box(temp_df,x="dvery_year",y="quantity tons")
    st.plotly_chart(fig)

    #) to get the data of quantity tons less than 0
    #temp_df[temp_df['quantity tons']<0]

    #) drop the data of quantity tons less than 0
    temp_df.drop([105730,131473,181661,181671],axis=0,inplace=True)

    st.markdown("\n#### :blue[2.10 Iteration 10:]\n")
    fig = px.box(temp_df,x="dvery_year",y="quantity tons")
    st.plotly_chart(fig)

    st.markdown("\n#### :blue[2.11 Iteration 11:]\n")
    fig = px.box(temp_df,x="dvery_year",y="country")
    st.plotly_chart(fig)

    st.markdown("\n#### :blue[2.12 Iteration 12:]\n")
    fig = px.box(temp_df,x="dvery_year",y="application")
    st.plotly_chart(fig)

    st.markdown("\n#### :blue[2.13 Iteration 13:]\n")
    fig = px.box(temp_df,x="dvery_year",y="thickness")
    st.plotly_chart(fig)

    #) checking outliers above 300 in thickness column
    #temp_df[temp_df['thickness']>300]

    #) dropping outliers above 300 in thickness column
    temp_df.drop([41,45001],axis=0,inplace=True)
    
    st.markdown("\n#### :blue[2.14 Iteration 14:]\n")
    fig = px.box(temp_df,x="dvery_year",y="thickness")
    st.plotly_chart(fig)

    st.markdown("\n#### :blue[2.15 Iteration 15:]\n")
    fig = px.box(temp_df,x="dvery_year",y="width")
    st.plotly_chart(fig)

    #) to check width column beyond 2200
    #temp_df[temp_df['width']>2200]

    #) dropping width column beyond 2200
    temp_df.drop([42120,58144,67018],axis=0,inplace=True)

    st.markdown("\n#### :blue[2.16 Iteration 16:]\n")
    fig = px.box(temp_df,x="dvery_year",y="width")
    st.plotly_chart(fig)

    #) to get the outlier in 2019
    #temp_df[(temp_df['dvery_year'] == 2019) & (temp_df['width']>1500)]

    #)to drop the outlier in 2019
    temp_df.drop(139368,axis=0,inplace=True)

    #) boxplot
    st.markdown("\n#### :blue[2.17 Iteration 17:]\n")
    fig = px.box(temp_df,x="dvery_year",y="width")
    st.plotly_chart(fig)

temp_df = pd.get_dummies(temp_df,['status','item type'])

#) 2.chart with radiobutton
st.subheader(":violet[3.Regression ML models ]")
RadioButton = st.radio(':blue[**Select the chart**]: ',('Linear',
                                             'Lasso',
                                             'Ridge'))
if RadioButton == 'Linear':
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    X = temp_df.drop(['selling_price'],axis=1)
    y = temp_df['selling_price']
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = LinearRegression()  
    model.fit(x_train,y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    st.write(f"\n:red[**{type(model).__name__}**]")
    st.success("\n:red[**Train data(MSE)**]")
    st.write(mean_squared_error(y_train,train_pred))
    st.error("\n:red[**Test data(MSE)**]")
    st.write(mean_squared_error(y_test,test_pred))

    #) actual testing vs testing prediction
    st.info("\n:red[**actual testing vs testing prediction**]")
    test_df = pd.DataFrame()
    test_df['test_actual']= y_test
    test_df['test_pred'] = test_pred
    st.dataframe(test_df.head(10))

elif RadioButton == 'Lasso':
    #)lasso
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    #temp_df = pd.get_dummies(temp_df,['status','item type'])
    X = temp_df.drop(['selling_price'],axis=1)
    y = temp_df['selling_price']
    
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model = Lasso()
    model.fit(x_train,y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    
    st.write(f"\n:red[**{type(model).__name__}**]")
    st.success("\n:red[**Train data(MSE)**]")
    st.write(mean_squared_error(y_train,train_pred))
    st.error("\n:red[**Test data(MSE)**]")
    st.write(mean_squared_error(y_test,test_pred))

    #) actual testing vs testing prediction
    st.info("\n:red[**actual testing vs testing prediction**]")
    test_df = pd.DataFrame()
    test_df['test_actual']= y_test
    test_df['test_pred'] = test_pred
    st.dataframe(test_df.head(10))

elif RadioButton == 'Ridge':
    #) ridge
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    #temp_df = pd.get_dummies(temp_df,['status','item type'])
    X = temp_df.drop(['selling_price'],axis=1)
    y = temp_df['selling_price']
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    model = Ridge()
    model.fit(x_train,y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    st.write(f"\n:red[**{type(model).__name__}**]")
    st.success("\n:red[**Train data(MSE)**]")
    st.write(mean_squared_error(y_train,train_pred))
    st.error("\n:red[**Test data(MSE)**]")
    st.write(mean_squared_error(y_test,test_pred))

    #) actual testing vs testing prediction
    st.info("\n:red[**actual testing vs testing prediction**]")
    test_df = pd.DataFrame()
    test_df['test_actual']= y_test
    test_df['test_pred'] = test_pred
    st.dataframe(test_df.head(10))