import streamlit as st

import text_preprocessing
import model

# title
st.title("Public Report Classification")

# load model, set cache to prevent reloading
#@st.cache(allow_output_mutation=True)
@st.cache_resource()
my_model = model.load_model()

# get new report
new_report = st.text_area("Ada masalah apa?","")

if st.button("Prediksi Kategori Laporan", use_container_width=True):
    with st.spinner("Tunggu sebentar, sedang memprediksi kategori laporan..."):
        df, preprocess_text = model.predict(my_model, new_report)
        if st.checkbox('Tampilkan teks hasil preprocessing'):
            st.write(preprocess_text)
        for cat, prob in zip(df['prediksi_kategori_laporan'], df['probability']):
            if prob < 5:
                break
            pred = cat + " | " + str(round(prob,2)) + "%"
            st.success(pred)
        st.divider() # draw a horizontal line
        if st.checkbox('Tampilkan semua kategori'):
            st.dataframe(df, use_container_width=True)
