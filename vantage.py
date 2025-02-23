from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import fit_model, predict_y


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """factor in price discount for auctions

    Args:
        data (pd.DataFrame): input data

    Returns:
        pd.DataFrame: output data
    """
    mask = data["source"] == "auction"
    data.loc[mask, "price"] = (
        data.loc[mask, "price"] + data.loc[mask, "price"] * 0.05 + 1200
    )

    return data


if __name__ == "__main__":

    # pandas read from excel file
    df = pd.read_excel("v8_vantage.xlsx")
    df = preprocess_data(df)

    features = ["miles"]
    categorical_columns = ["packages"]
    output_column = "price"
    encoder, model = fit_model(
        df, features, output_column, categorical_columns=categorical_columns
    )

    # prediction
    df_cb = pd.DataFrame.from_dict({'miles': [44300], 'packages': ['manual']})
    y_cb = predict_y(df_cb, features, model, categorical_columns=categorical_columns, encoder=encoder)
    adjusted_y_cb = y_cb - y_cb * 0.05 - 1200
    print('Predicted Price of Cars and Bids Car: ${:,.2f}'.format(adjusted_y_cb[0,0]))
    milage_new = np.linspace(10000, 90000, 200)
    dd_manual = {"miles": milage_new, "packages": ["manual"] * len(milage_new)}
    dd_automatic = {"miles": milage_new, "packages": ["automatic"] * len(milage_new)}
    df_manual = pd.DataFrame.from_dict(dd_manual)
    df_auto = pd.DataFrame.from_dict(dd_automatic)

    y_manual = predict_y(
        df_manual,
        features,
        model,
        categorical_columns=categorical_columns,
        encoder=encoder,
    )
    y_auto = predict_y(
        df_auto,
        features,
        model,
        categorical_columns=categorical_columns,
        encoder=encoder,
    )

    # Plot
    manual_mask = df["packages"] == "manual"
    plt.figure()
    plt.plot(milage_new, y_manual, "-r")
    plt.plot(df[manual_mask]["miles"], df[manual_mask]["price"], "ob")
    plt.xlabel("Milage [Miles]")
    plt.ylabel("Price [$]")
    plt.title("V8 Vantage Price Analysis\nManual")
    plt.grid()

    manual_mask = df["packages"] == "automatic"
    plt.figure()
    plt.plot(milage_new, y_auto, "-r")
    plt.plot(df[manual_mask]["miles"], df[manual_mask]["price"], "ob")
    plt.xlabel("Milage [Miles]")
    plt.ylabel("Price [$]")
    plt.title("V8 Vantage Price Analysis\nAutomatic")
    plt.grid()
    plt.show()
