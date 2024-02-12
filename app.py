import streamlit as st

st.title("TEST APP")

ticker_list = ["SP500", "NASDAQ"]
st.selectbox("Select asset", ticker_list)

n_years = st.slider("Years", 1, 4)

# st.write(Dataframe.head(5))
# st.plotly_chart(fig)