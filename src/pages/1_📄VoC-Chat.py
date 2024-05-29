import os
import streamlit as st
from io import StringIO
import re
import sys
from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar

# To be able to update the changes made to modules in localhost (press r)
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]

history_module = reload_module('modules.history')
layout_module = reload_module('modules.layout')
utils_module = reload_module('modules.utils')
sidebar_module = reload_module('modules.sidebar')

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar

st.set_page_config(layout="wide", page_icon="💬", page_title="AI Chatbot 🤖 Voice Of Customer")

# 实例化主要组件
layout, sidebar, utils = Layout(), Sidebar(), Utilities()

layout.show_header("PDF, TXT, CSV")


user_api_key = utils.load_api_key()

if not user_api_key:
    layout.show_api_key_missing()
else:
    os.environ["OPENAI_API_KEY"] = user_api_key

    uploaded_file = utils.handle_upload(["pdf", "txt", "csv", "xlsx"])

    if uploaded_file:
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['ready'] = True

    if st.session_state['uploaded_file']:
        uploaded_file = st.session_state['uploaded_file']

        # 配置侧边栏
        sidebar.show_options()
        sidebar.about()

        # 初始化聊天记录
        history = ChatHistory()

        # 从 session state 加载聊天记录
        if st.session_state['chat_history']:
            for entry in st.session_state['chat_history']:
                history.append(entry['mode'], entry['message'])

        try:
            chatbot = utils.setup_chatbot(
                uploaded_file, st.session_state.get("model", "default-model"), st.session_state.get("temperature", 0.7)
            )
            st.session_state["chatbot"] = chatbot

            if st.session_state["ready"]:
                # 创建用于聊天响应和用户提示的容器
                response_container, prompt_container = st.container(), st.container()

                with prompt_container:
                    # 显示提示表单
                    is_ready, user_input = layout.prompt_form()

                    # 初始化聊天记录
                    history.initialize(uploaded_file)

                    # 重置聊天记录按钮点击时重置聊天记录
                    if st.session_state["reset_chat"]:
                        history.reset(uploaded_file)
                        st.session_state['chat_history'] = []
                        st.session_state["reset_chat"] = False

                    if is_ready:
                        # 更新聊天记录并显示聊天消息
                        history.append("user", user_input)
                        st.session_state['chat_history'].append({"mode": "user", "message": user_input})

                        old_stdout = sys.stdout
                        sys.stdout = captured_output = StringIO()

                        output = st.session_state["chatbot"].conversational_chat(user_input)

                        sys.stdout = old_stdout

                        history.append("assistant", output)

                        # 保存聊天历史到 session state 中
                        st.session_state['chat_history'].append({"mode": "assistant", "message": output})

                        # 清理代理的思路以删除不需要的字符
                        thoughts = captured_output.getvalue()
                        cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                        cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)

                        # 显示代理的思路
                        with st.expander("Display the agent's thoughts"):
                            st.write(cleaned_thoughts)

                history.generate_messages(response_container)

        except Exception as e:
            st.error(f"Error: {str(e)}")
