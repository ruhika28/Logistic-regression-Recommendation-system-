import  numpy as np
import  pandas as pd
from pandas import Series,DataFrame
from sklearn.linear_model import LogisticRegression
from tkinter.filedialog import askopenfilename
from sklearn.metrics import classification_report

print (sys.version)
fileloc=askopenfilename()
bank_data=pd.read_csv(fileloc)
bank_data.info()
#binary  values  taken as input for factor to make the prediction
x=bank_data.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
#prediction variable
y=bank_data.ix[:,17].values

#logisteic regression
logreg=LogisticRegression()
logreg.fit(x,y)
#create the array that defines the user based on characteristics
new_user_sam = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]).reshape(1,-1)
#make the prediction wheteher  he will take or not taske the marketted scheme
y_prediction=logreg.predict(new_user_sam)
print("sams prediction:")
if (y_prediction==0):
      print("Sams not going to buy the product ")
else:
      print("sam is going to  buy the bank recommended product")
#op is  0 suggesting that user wont purchase the scheme

#now we have  to observe how accurate the  prediction made is wherein the predicted variable is y that tells us how
# relevant the recommendations were


ypred2=logreg.predict(x)
print(classification_report(y,ypred2))
sh=input("enter stuff:")
print(list(sh))