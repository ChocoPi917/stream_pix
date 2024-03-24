import streamlit as st
st.write("入力し終えたらEnterキーを押してください")
sentences=st.text_input("文字数を数えたい文章を入力")
st.write("文字数",len(sentences))