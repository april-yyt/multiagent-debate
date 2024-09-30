import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from agent.app import create_workflow, run_workflow

# Streamlit UI
st.set_page_config(page_title="Multiagent Debate Framework", page_icon=":speech_balloon:", layout="wide")

# Initialize the language models
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def get_claude():
    return ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
    )

llm = get_llm()
claude = get_claude()

st.title("Multiagent Debate Framework")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "waiting" not in st.session_state:
    st.session_state.waiting = False
if "current_task" not in st.session_state:
    st.session_state.current_task = ""
if "debate_log" not in st.session_state:
    st.session_state.debate_log = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if not st.session_state.waiting:
    user_input = st.chat_input("Enter your task or question:")
else:
    user_input = st.chat_input("Agents are debating...", disabled=True)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.waiting = True
    st.session_state.current_task = user_input
    st.rerun()

if st.session_state.waiting and st.session_state.current_task:
    with st.chat_message("assistant"):
        with st.spinner("Agents are debating..."):
            # Create containers for each agent node
            planner_container = st.expander("Planner - CoT A", expanded=False)
            replanner_container = st.expander("Replanner - CoT B", expanded=False)
            final_container = st.expander("Final Answer", expanded=False)

            # Create workflow app
            app = create_workflow(llm, claude)

            try:
                for state in run_workflow(app, {"task": st.session_state.current_task}):
                    if "cot_a" in state:
                        plan_content = state["cot_a"]
                        planner_container.write(plan_content["cot_a_result"])

                    if "cot_b" in state:
                        replanner_content = state["cot_b"]
                        replanner_container.write(replanner_content["cot_b_result"])

                    if "final_answer" in state:
                        final_content = state["final_answer"]
                        final_container.write(final_content["final_answer"])
                        st.session_state.messages.append({"role": "assistant", "content": final_content["final_answer"]})
                        break

                st.session_state.waiting = False
                st.session_state.current_task = ""

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.waiting = False
                st.session_state.current_task = ""

    # st.rerun()