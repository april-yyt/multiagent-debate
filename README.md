# **Multiagent Debate Framework for Inference Quality Improvement**

## **Introduction**

This repository implements a novel approach to improving the quality of language model inference through a **Multiagent Debate Framework**. The system dynamically balances between fast, concise reasoning and more detailed, thoughtful deliberation by combining multiple agents in a flexible architecture.

The framework is designed to optimize for both **quality** and **efficiency**. It utilizes a multi-stage process to determine whether a task can be handled by quick inference or if it requires more comprehensive reasoning. By integrating various agents for planning, re-evaluating, and triggering deeper reasoning processes, this framework aims to enhance inference quality without sacrificing speed.

### **Key Components:**

1. **Planner**:  
   * Breaks down the task into reasoning steps, generating a task list for further processing.  
2. **Replanner**:  
   * Evaluates the output of the initial agent (CoT A) and decides whether further reasoning is required. It triggers a more detailed reasoning agent (CoT B) if necessary.  
3. **Trigger Model**:  
   * Acts as the decision-making component to trigger the longer reasoning chain (CoT B). It utilizes simple classifiers like Decision Trees, Random Forests, or lightweight transformers trained on synthetic data.  
4. **Benchmarking Dataset**:  
   * The framework uses **GSM8K**, a dataset of math word problems, to test the reasoning capabilities of the system.

---

## **Installation**

### **1\. Clone the Repository**

```
git clone https://github.com/april-yyt/multiagent-debate.git 
cd multiagent-debate
```

### **2\. Set Up Virtual Environment**

Create and activate a Python virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

### **3\. Install Required Dependencies**

Use the `requirements.txt` file to install all the necessary Python libraries:

```
pip install -r requirements.txt
```

### **4\. API Keys Setup**

To use language models like OpenAI and Anthropic, you need to set up API keys. Store these keys in a `.streamlit/secrets.toml` file like this:

```toml  
OPENAI_API_KEY = "your-openai-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

---

## **How to Run**

### **1\. Running the App**

Use the provided shell script `run_app.sh` to set up your environment variables and launch the Streamlit app.


```
chmod +x run_app.sh
./run_app.sh
```

Alternatively, you can manually run the app using:

```
streamlit run sl/multiagent_debate_app.py
```

### **2\. What to Expect**

Once the application is running, you'll interact with a Streamlit interface where you can input a task or question. The agents will debate the task and return a final answer after dynamically balancing between fast inference and more detailed reasoning.

---

## **Project Structure**

```
multiagent-debate-framework/
│
├── agent/
│   └── nodes/
│       ├── cot_long.py         # Detailed reasoning (CoT B)
│       ├── cot_short.py        # Concise reasoning (CoT A)
│       ├── planner.py          # Task breakdown planner
│       ├── replanner.py        # Evaluates necessity for more reasoning
│       ├── summarizer.py       # Generates the final answer
│       ├── trigger_model.py    # Determines if detailed reasoning is needed
│       └── __init__.py
│
├── sl/
│   └── multiagent_debate_app.py # Streamlit UI code
│
├── venv/                        # Virtual environment directory
├── requirements.txt             # Python dependencies
├── LICENSE                      # License file
├── run_app.sh                   # Script to run the Streamlit app
└── README.md                    # This file
```
---

