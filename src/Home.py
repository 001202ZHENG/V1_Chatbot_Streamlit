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
    <h2 style='text-align: center;'>VoC, where your voice is heard and advice is served. 🤖</h1>
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


'''#Robby's Pages
st.subheader("🚀 Robby's Pages")
st.write("""
- **Robby-Chat**: General Chat on data (PDF, TXT,CSV) with a [vectorstore](https://github.com/facebookresearch/faiss) (index useful parts(max 4) for respond to the user) | works with [ConversationalRetrievalChain](https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html)
- **Robby-Sheet** (beta): Chat on tabular data (CSV) | for precise information | process the whole file | works with [CSV_Agent](https://python.langchain.com/en/latest/modules/agents/toolkits/examples/csv.html) + [PandasAI](https://github.com/gventuri/pandas-ai) for data manipulation and graph creation
- **Robby-Youtube**: Summarize YouTube videos with [summarize-chain](https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html)
""")
st.markdown("---")
'''

#Contributing
st.markdown("### 🎯 Contribution")
st.markdown("""
**This chatbot is developed through the collaborative efforts of the Voice of Customers teams from ESSEC Business School & CentraleSupélec, consisting of Data Sciences & Business Analytics students, along with the Deloitte Team. 🚀**
""", unsafe_allow_html=True)





