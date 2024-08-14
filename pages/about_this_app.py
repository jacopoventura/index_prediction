import streamlit as st

st.markdown('<h2 style="text-align: center;">About this app</h2>', unsafe_allow_html=True)

st.markdown('<div style="text-align: justify">This application predicts the price of a stock in the next trading day(s). The output is '
            'either the predicted price or the change (positive/negative).</div>', unsafe_allow_html=True)

st.markdown('<div> </div> ', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;">Details</h3>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify">The app runs a trained deepl learning model capable to predict the price of the stock.</div>',
            unsafe_allow_html=True)
st.markdown('<div style="text-align: justify">Since the 01.01.2020, the market (SP500) featured 53% of positive days. </div>', unsafe_allow_html=True)

st.markdown('<div> </div> ', unsafe_allow_html=True)
st.markdown('<div> </div> ', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center;">Copyright</h2>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center">Copyright (c) Jacopo Ventura, 2024. Distribution not allowed.</div>', unsafe_allow_html=True)
