from langchain.prompts import ChatPromptTemplate

def replanner(state, llm):
    print("Replanner input state:", state)  # Debug print
    prompt = ChatPromptTemplate.from_template(
        "Task: {task}\nInitial reasoning: {cot_a_result}\n\nEvaluate if further reasoning is necessary. Respond with 'Yes', 'No', or 'Need Human Input'."
    )
    response = llm.invoke(prompt.format_messages(task=state["task"], cot_a_result=state["cot_a_result"]))
    decision = response.content.strip().lower()

    if decision == "yes":
        state["needs_further_reasoning"] = True
    elif decision == "no":
        state["needs_further_reasoning"] = False
    else:  # "need human input"
        state["needs_human_input"] = True

    print("Replanner output state:", state)  # Debug print
    return state
