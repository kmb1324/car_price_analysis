from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pandas read from excel file
df = pd.read_csv('c5_z06.csv')


milage_new = np.linspace(20000, 150000, 200)
price_117000 = df[df['miles'] == 117000].iloc[0]['price']


adjusted_price = df['price'].values
fit = LinearRegression().fit(df['miles'].values.reshape(-1,1), adjusted_price)
price_new = fit.predict(milage_new.reshape(-1,1))
price_facebook = fit.predict(np.array([117000]).reshape(1,-1))[0]
sandy_price = fit.predict(np.array([100000]).reshape(1,-1))[0]

discount_sandys = (price_facebook - price_117000)/price_facebook

plt.figure()
plt.plot(df['miles'].values,adjusted_price,'ob')
plt.plot(milage_new,price_new,'-r')
plt.xlabel('Milage [Miles]')
plt.ylabel('Price [$]')
plt.title('C5 Z06 Price Analysis\nPredicted Price (w/ discount): ${:.2f}'.format(sandy_price*(1-discount_sandys)))
plt.legend(['Facebook For Sale', 'Predicted Price (Regression)'])
plt.grid()
plt.show()
