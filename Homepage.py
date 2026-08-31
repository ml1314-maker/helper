# ================================================================
# Python · 智能编程助手
# 适用对象：初中生（编程基础薄弱）
# 特点：启发式引导、不直接给答案、鼓励试错
# ================================================================

import openai
import streamlit as st
import pandas as pd
from datetime import datetime
import re
import os


# ================================================================
# 1. OpenAI客户端配置
# ================================================================

def setup_openai_client(api_key, service="qwen"):
    """配置OpenAI客户端，支持千问和DeepSeek两种服务"""
    base_urls = {
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "deepseek": "https://api.deepseek.com/v1"
    }
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_urls[service]
    )


def get_api_response(messages, temperature, apis, models, service="qwen"):
    """尝试多个API密钥和模型获取响应"""
    for model in models:
        for api in apis:
            try:
                client = setup_openai_client(api, service)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=temperature
                )
                return response, False
            except Exception as e:
                st.error(f"连接失败，正在尝试下一个...")
                continue
    return None, True


# ================================================================
# 2. 编程助手教育提示语（仅在代码中，不展示在侧边栏）
# ================================================================

PROGRAMMING_ASSISTANT_PROMPT = """
# 背景
通用大语言模型不足以满足初中生编程入门教学中的个性化指导需求，因此设计编程教学智能体，旨在通过苏格拉底式启发对话，帮助学生在编程学习过程中克服困难，培养独立解决问题的能力。本课程面向初中七年级至八年级学生，编程基础薄弱，对代码可能存在畏难情绪，需要通过启发式引导降低认知负荷，逐步建立编程自信。

# 动机
研发编程教学智能体，需要针对智能体设定合适的教育提示语，从而使得学生在编程学习过程中能够获得高质量的启发式指导，真正实现"做中学、问中思"，助力学生编程能力与计算思维的双重提升。

# 结构和关键点
背景情况：本课程《Python趣味编程》以turtle海龟库为载体，通过趣味游戏项目（如"打地鼠""猫鼠游戏""彩色螺旋"等）引导学生学习Python编程。智能体作为课程中的"苏格拉底式导师"，在学生编程学习过程中提供个性化启发与引导，而非直接告知答案。

角色功能：你是一位专业、耐心、友善的编程导师，专门辅导初中生学习Python编程。
① 启发式引导：学生在编程过程中遇到困难时，通过提问引导他们分析问题、拆解步骤、寻找解决方案，绝对禁止直接给出完整代码。
② 开放式提问：用"你觉得下一步该做什么？""如果换成……会怎样？""这段代码的运行结果可能是什么？"等开放式问题，引导学生独立思考。
③ 生活化类比：用学生熟悉的生活场景解释抽象编程概念（如"变量就像贴了名字的收纳盒""循环就像体育课上的报数"）。
④ 分解式教学：将复杂任务拆解为若干小步骤，一步步引导学生完成，每次只聚焦一个小目标。
⑤ 只回应编程问题：对于非编程问题，礼貌地引导学生提出编程相关的问题。

目标：你的最终目标是通过启发式对话，帮助学生建立编程自信、培养计算思维、提升问题解决能力。在与学生互动时，应做到：多用开放式提问激发思考；及时肯定学生的每一点进步（如"太好了！""你做到了！"）；绝对不说"这很简单""你应该这样做"等打击学生积极性的话语。你只回应编程类问题，如遇不相关提问，则引导用户提出编程相关内容。
"""


# ================================================================
# 3. 交互频率统计函数
# ================================================================

def analyze_interaction_frequency(messages):
    """
    分析聊天记录的交互频率
    返回：总轮数、用户发言次数、助手发言次数、平均消息长度等
    """
    user_msgs = []
    assistant_msgs = []

    for msg in messages:
        if msg["role"] == "user":
            user_msgs.append(msg["content"])
        elif msg["role"] == "assistant":
            assistant_msgs.append(msg["content"])

    total_rounds = len(user_msgs)

    avg_user_len = sum(len(m) for m in user_msgs) / len(user_msgs) if user_msgs else 0
    avg_assistant_len = sum(len(m) for m in assistant_msgs) / len(assistant_msgs) if assistant_msgs else 0

    question_keywords = {
        "概念": ["什么是", "是什么", "什么意思", "解释", "概念"],
        "调试": ["报错", "错误", "不行", "不对", "bug", "调试", "运行不了"],
        "思路": ["怎么", "如何", "怎样", "思路", "方法", "步骤"],
        "代码": ["代码", "写", "实现", "完成"]
    }

    question_types = {k: 0 for k in question_keywords}
    for msg in user_msgs:
        for qtype, keywords in question_keywords.items():
            if any(kw in msg for kw in keywords):
                question_types[qtype] += 1

    return {
        "总交互轮数": total_rounds,
        "用户提问次数": len(user_msgs),
        "助手回复次数": len(assistant_msgs),
        "用户平均提问长度": round(avg_user_len, 1),
        "助手平均回复长度": round(avg_assistant_len, 1),
        "问题类型统计": question_types,
        "用户完整消息": user_msgs,
        "助手完整消息": assistant_msgs
    }


# ================================================================
# 4. 主程序
# ================================================================

def main():
    st.set_page_config(
        page_title="Python · 智能编程助手",
        page_icon="🐢",
        layout="wide"
    )

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.markdown("""
        <center>
        <h1>🐢 智能编程助手</h1>
        <p style="color: #666; font-size: 14px;">苏格拉底式启发 · 引导你思考</p>
        </center>
        """, unsafe_allow_html=True)

        st.divider()

        # 学习提示单（简洁版）
        st.markdown("""
        ## 📋 学习提示单

        **🎯 提问公式：目的 ＋ 怎么做 ＋ 输出格式 ＋ 示例**

        | 组成 | 含义 | 案例 |
        |------|------|------|
        | **目的** | 你想解决什么问题？ | 我想用turtle画五角星，但不知道角度怎么算。 |
        | **怎么做** | 你希望助手怎么帮你？ | 用生活例子解释，分步骤引导，不给完整代码。 |
        | **输出格式** | 你希望答案是什么样的？ | 像朋友聊天，多用"你觉得呢？"，加表情符号 🐢。 |
        | **示例** | 参考例子 | **人**：turtle怎么画彩色螺旋？<br>**助手**：画螺旋需要重复做什么动作？ |

        ⭐ **记住：助手不会直接给答案，会引导你自己想出来！**
        """, unsafe_allow_html=True)

        st.divider()

        # 创造力调节
        temperature = st.slider(
            "🎨 创造力调节",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="值越大回答越灵活多样，值越小回答越稳定保守"
        )

        # 保存聊天记录
        st.divider()

        if st.button("💾 保存聊天记录"):
            if "messages" in st.session_state:
                messages = st.session_state["messages"]
                chat_messages = [msg for msg in messages if msg["role"] != "system"]

                if len(chat_messages) == 0:
                    st.warning("⚠️ 暂无聊天记录可保存")
                else:
                    # 保存聊天记录
                    data = []
                    for msg in chat_messages:
                        data.append({
                            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "角色": "我" if msg["role"] == "user" else "助手",
                            "内容": msg["content"]
                        })
                    df = pd.DataFrame(data)
                    filename = f"聊天记录_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    df.to_excel(filename, index=False)

                    st.success(f"✅ 聊天记录已保存，点击下方按钮下载")

                    # 下载按钮（直接下载到桌面）
                    with open(filename, "rb") as f:
                        st.download_button(
                            label="📥 点击下载到桌面",
                            data=f,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.warning("⚠️ 暂无聊天记录")

        # 清空对话
        if st.button("🗑️ 清空对话"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": welcome_message()}
            ]
            st.rerun()

    # ---------- 主界面 ----------
    st.title("🐢 Python · 智能编程助手")
    st.caption("我不会直接给你答案，但我会引导你自己找到答案 💡")

    # 初始化消息历史
    def welcome_message():
        return """👋 你好！我是你的Python编程导师～

我不会直接给你写代码，但我会**引导你思考**，让你自己找到答案！

🐢 你可以问我：
- 💡 概念理解（如"什么是变量？"）
- 🐛 代码调试（如"我的代码报错了"）
- 🎯 编程思路（如"怎么画一个五角星？"）

记住：**自己找到的答案，才是真正学会的！** 我们一起加油！ 💪"""

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": welcome_message()}
        ]

    # 显示聊天历史
    for msg in st.session_state["messages"]:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ---------- 核心对话函数 ----------
    def chat_with_ai(user_input, temperature=0.7):
        """调用AI，流式返回响应"""

        messages = [
            {"role": "system", "content": PROGRAMMING_ASSISTANT_PROMPT}
        ]

        recent = st.session_state["messages"][-10:] if len(st.session_state["messages"]) > 10 else st.session_state[
            "messages"]
        messages.extend(recent)
        qwen_apis = [
            os.getenv("QWEN_API_KEY_1"),
            os.getenv("QWEN_API_KEY_2"),
            os.getenv("QWEN_API_KEY_3"),
            os.getenv("QWEN_API_KEY_4")
        ]
        qwen_models = ["qwen-turbo", "qwen-plus", "qwen-max", "qwq-plus"]

        response, error = get_api_response(messages, temperature, qwen_apis, qwen_models, "qwen")
        if not error:
            return response
        deepseek_apis = [
                    os.getenv("DEEPSEEK_API_KEY_1"),
                    os.getenv("DEEPSEEK_API_KEY_2")
                ]
        deepseek_models = ["deepseek-chat"]

        response, error = get_api_response(messages, temperature, deepseek_apis, deepseek_models, "deepseek")
        if error:
            st.error("😅 哎呀，AI暂时睡着了，请稍后再试～")
            return None

        return response

    # ---------- 用户输入 ----------
    user_input = st.chat_input("输入你的问题，让我引导你思考...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            response = chat_with_ai(user_input, temperature)

            if response is not None:
                placeholder = st.empty()
                full_response = ""

                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)
                st.session_state["messages"].append({"role": "assistant", "content": full_response})


# ================================================================
# 5. 启动应用
# ================================================================

if __name__ == "__main__":
    main()
