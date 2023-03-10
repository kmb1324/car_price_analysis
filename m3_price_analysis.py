from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pandas read from excel file
df = pd.read_excel('e92_auctions.xlsx')
print(df)

milage = np.array([42000,
41300,
13800,
74500,
46064,
13316,
50955,
52117,
20887
])


price = np.array([44869, 
40550, 
50750,
54000, 
54950, 
65500, 
46999, 
54799, 
73240 
])

milage_new = np.linspace(10000, 90000, 200)
milage92 = np.linspace(20000, 100000, 200)

fit = LinearRegression().fit(milage.reshape(-1,1), price)
fit92 = LinearRegression().fit(df['miles'].values.reshape(-1,1), df['price'].values + df['price'].values*0.05)

price_new = fit.predict(milage_new.reshape(-1,1))
price92_new = fit92.predict(milage92.reshape(-1,1))
price_sunny = fit92.predict(np.array([68900]).reshape(1,-1))

plt.figure()
plt.plot(milage_new,price_new,'-r')
plt.plot(milage,price,'ob')
plt.xlabel('Milage [Miles]')
plt.ylabel('Price [$]')
plt.title('F80 M3 Price Analysis')
plt.grid()

plt.figure()
plt.plot(milage92,price92_new,'-r')
plt.plot(df['miles'].values,df['price'].values,'ob')
plt.xlabel('Milage [Miles]')
plt.ylabel('Price [$]')
plt.title('E92 M3 Price Analysis\n68k Miles Predicts a Price of ${0:.2f}'.format(price_sunny[0]))
plt.grid()


print('Price for 68900 miles: ${:.2f}'.format(price_sunny[0]))

plt.show()
