# %%
import streamlit as st
import pandas as pd
from prophet import Prophet

st.title("Prophet Forecast Tool")

file = st.file_uploader("Upload EXCEL", type="xlsx")

if file:
    df = pd.read_excel(file)
    st.dataframe(df.head())

    sku_col = st.selectbox("SKU column", df.columns)
    ds_col = st.selectbox("Date column", df.columns)
    y_col = st.selectbox("Target column", df.columns)

    sku_options = ["All SKUs"] + sorted(df[sku_col].unique())
    sku_value = st.selectbox("Select SKU", sku_options)

    horizon = st.number_input("Forecast days", 7, 365, 30)
    cps = st.slider("Changepoint prior scale", 0.01, 0.5, 0.05)

    if st.button("Run forecast"):
        if sku_value == "All SKUs":
            grouped = df.groupby(ds_col)[y_col].sum().reset_index()
            data = grouped.rename(columns={ds_col: "ds", y_col: "y"})
            st.subheader("All SKUs combined")
        else:
            data = df[df[sku_col] == sku_value][[ds_col, y_col]]
            data = data.rename(columns={ds_col: "ds", y_col: "y"})
            st.subheader(f"SKU: {sku_value}")

        data["ds"] = pd.to_datetime(data["ds"])

        model = Prophet(changepoint_prior_scale=cps)
        model.fit(data)

        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        forecast["yhat"] = forecast["yhat"].round().astype(int)
        forecast["yhat_lower"] = forecast["yhat_lower"].round().astype(int)
        forecast["yhat_upper"] = forecast["yhat_upper"].round().astype(int)
        st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
        st.dataframe(forecast.tail(horizon))
