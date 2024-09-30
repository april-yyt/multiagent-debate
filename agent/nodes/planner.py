from langchain.prompts import ChatPromptTemplate

def planner(state, llm):
    print("Planner input state:", state)  # Debug print
    prompt = ChatPromptTemplate.from_template(
        "Given the task: {task}, break it down into a list of reasoning steps."
    )
    response = llm.invoke(prompt.format_messages(task=state["task"]))
    state["plan"] = response.content
    print("Planner output state:", state)  # Debug print
    return state
