import streamlit as st

import text_preprocessing
import model

# Title
st.title("Public Report Classification")

# Load model
with st.spinner("Loading Model...."):
    my_model = model.load_model()

# Get new report
new_report = st.text_area("Ada masalah apa?","")

# Define the CSS style for the cards
card_style = """
    background-color: #f5f5f5;
    border-radius: 10px;
    box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.2);
    padding: 20px;
    margin: 10px;
    height: 200px;
    display: flex;
    flex-direction: column;
"""

if st.button("Prediksi Kategori Laporan", use_container_width=True):
    with st.spinner("Tunggu sebentar, sedang memprediksi kategori laporan..."):
        df, preprocess_text = model.predict(my_model, new_report)
        
        col1, col2, col3 = st.columns(3)
        for index, row in df.iterrows():
            if index % 3 == 0:
                card_col = col1
            elif index % 3 == 1:
                card_col = col2
            else:
                card_col = col3
            with card_col:
                st.markdown(
                    f"""
                    <div style="{card_style}">
                        <h3>{row['prediksi_kategori_laporan']}</h3>
                        <p>Akurasi: {row['probability']}</p>
                        <p>Deskripsi: [Masukan deskripsi laporan disini]</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
        
    st.divider() # Draw a horizontal line
    
    # Show preprocessed text
    with st.expander("Teks hasil preprocessing"):
        st.write(preprocess_text)
        
    # Show table of all categories
    with st.expander("Tabel semua prediksi kategori"):
        st.dataframe(df, use_container_width=True)
