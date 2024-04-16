import streamlit as st


#Config
st.set_page_config(layout="wide", page_icon="💬", page_title="AI Chatbot 🤖 Voice Of Customer")


#Contact
with st.sidebar.expander("📬 Contact"):

    st.write("**GitHub:**",
"[V1_Chatbot_Streamlit](https://github.com/001202ZHENG/V1_Chatbot_Streamlit)")
    st.write("**Mail** :", "zheng.wan@student-cs.fr [Deloitte Team]")
    st.write("**Mail** :", "TNuss@deloitte.fr [Deloitte Team]")
    st.write("** By “Voice of Customers” Team**")


#Title
st.markdown(
    """
    <h2 style='text-align: center;'>VoC, where your voice is heard and advice is served! 🤖</h1>
    """,
    unsafe_allow_html=True,)

st.markdown("---")


#Description
st.markdown(
    """ 
    <h5 style='text-align:center;'>Hey there! I'm VoC, your friendly chatbot here to lend an ear to your HR service needs and dish out some heartfelt advice. 🤖 I'm all about using cutting-edge language models to give you spot-on insights and make navigating your HR services a breeze. Let's work together to iron out any wrinkles in your HR processes! 💼 I'm your go-to guy for handling PDFs, TXTs, and CSV files. Let's chat! 🧠</h5>
    """,
    unsafe_allow_html=True)
st.markdown("---")



#Contributing
st.markdown("### 🎯 Contribution")
st.markdown("""
**This chatbot is developed through the collaborative efforts of the Voice of Customers teams from ESSEC Business School & CentraleSupélec, along with the Deloitte Team. 🚀**
""", unsafe_allow_html=True)





