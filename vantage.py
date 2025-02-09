from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pandas read from excel file
df = pd.read_excel('v8_vantage.xlsx')
X = df[['miles', 'packages']]

# Preprocess
encoder = OneHotEncoder(sparse_output=False)
X_transformed = encoder.fit_transform(X[['packages']])
X_train = np.hstack((df['miles'].values.reshape(-1,1), X_transformed))
print(X_train)

# prediction
milage_new = np.linspace(10000, 90000, 200)
dd_manual = {
    'miles': milage_new,
    'packages': ['manual'] * len(milage_new)
}
dd_automatic = {
    'miles': milage_new,
    'packages': ['automatic'] * len(milage_new)
}
df_predict_manual = pd.DataFrame.from_dict(dd_manual)
df_predict_auto = pd.DataFrame.from_dict(dd_automatic)
predict_transformed_manual = encoder.transform(df_predict_manual[['packages']])
predict_transformed_auto = encoder.transform(df_predict_auto[['packages']])
X_predict_manual = np.hstack((df_predict_manual['miles'].values.reshape(-1,1), predict_transformed_manual))
X_predict_auto = np.hstack((df_predict_manual['miles'].values.reshape(-1,1), predict_transformed_auto))

mask = df['source'] == 'auction'
df.loc[mask,'price'] =  df.loc[mask, 'price'] + df.loc[mask, 'price']*0.05 + 1200

fit = LinearRegression().fit(X_train, df['price'].values.reshape(-1,1))
price_predicted_manual = fit.predict(X_predict_manual)
price_predicted_auto = fit.predict(X_predict_auto)

manual_mask = df['packages'] == 'manual'
plt.figure()
plt.plot(milage_new, price_predicted_manual,'-r')
plt.plot(df[manual_mask]['miles'], df[manual_mask]['price'], 'ob')
plt.xlabel('Milage [Miles]')
plt.ylabel('Price [$]')
plt.title('V8 Vantage Price Analysis\nManual')
plt.grid()


manual_mask = df['packages'] == 'automatic'
plt.figure()
plt.plot(milage_new, price_predicted_auto,'-r')
plt.plot(df[manual_mask]['miles'], df[manual_mask]['price'], 'ob')
plt.xlabel('Milage [Miles]')
plt.ylabel('Price [$]')
plt.title('V8 Vantage Price Analysis\nAutomatic')
plt.grid()
plt.show()
