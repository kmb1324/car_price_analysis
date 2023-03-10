from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pandas read from excel file
df = pd.read_excel('997_auctions.xlsx')


milage_new = np.linspace(10000, 90000, 200)


adjusted_price = df['price'].values + df['price'].values*0.05 + 1200
fit = LinearRegression().fit(df['miles'].values.reshape(-1,1), adjusted_price)
price_new = fit.predict(milage_new.reshape(-1,1))
price_facebook = fit.predict(np.array([81000]).reshape(1,-1))

plt.figure()
plt.plot(milage_new,price_new,'-r')
plt.plot(df['miles'].values,adjusted_price,'ob')
plt.xlabel('Milage [Miles]')
plt.ylabel('Price [$]')
plt.title('Porche 997 Price Analysis\nFacebook Predicted Price: ${:.2f}'.format(price_facebook[0]))
plt.grid()
plt.show()
