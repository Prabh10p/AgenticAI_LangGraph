
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
import operator

# Load environment variables
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",  # ✅ this one exists
    task="conversational"
)


model = ChatHuggingFace(llm=llm)


class md(BaseModel):
    feedback: str = Field(description="Provide detailed feedback on the essay.")
    score: int = Field(description="Rate the essay out of 10.", ge=0, le=10)

structured_output = model.with_structured_output(md)


# Define Essay State

class EssayState(TypedDict):
    essay: str
    clarity_analysis: str
    depth_analysis: str
    grammar_analysis: str
    feedback: str
    score: Annotated[List[int], operator.add]
    avg_score: float


# Define Node Functions

def clarity_analysis(state: EssayState):
    essay = state["essay"]
    prompt = PromptTemplate(
        template=(
            "Evaluate the clarity of thought in the following essay:\n\n{essay}\n\n"
            "Provide detailed feedback and rate it out of 10."
        ),
        input_variables=["essay"]
    )
    result = structured_output.invoke(prompt)
    return {"clarity_analysis": result.feedback, "score": [result.score]}


def depth_analysis(state: EssayState):
    essay = state["essay"]
    prompt = PromptTemplate(
        template=(
            "Evaluate the depth of analysis in the following essay:\n\n{essay}\n\n"
            "Provide detailed feedback and rate it out of 10."
        ),
        input_variables=["essay"]
    )
    result = structured_output.invoke(prompt)
    return {"depth_analysis": result.feedback, "score": [result.score]}


def grammar_analysis(state: EssayState):
    essay = state["essay"]
    prompt = PromptTemplate(
        template=(
            "Evaluate the grammar and language quality of the following essay:\n\n{essay}\n\n"
            "Provide detailed feedback and rate it out of 10."
        ),
        input_variables=["essay"]
    )
    result = structured_output.invoke(prompt)
    return {"grammar_analysis": result.feedback, "score": [result.score]}


def summary(state: EssayState):
    essay = state["essay"]
    avg_score = sum(state["score"]) / len(state["score"])
    prompt = PromptTemplate(
        template=(
            "Summarize the evaluation of the essay below, integrating clarity, depth, "
            "and grammar feedback. Provide final overall feedback.\n\n{essay}\n\n"
            f"Average Score: {avg_score:.2f}/10"
        ),
        input_variables=["essay"]
    )
    result = structured_output.invoke(prompt)
    return {"feedback": result.feedback, "avg_score": avg_score}


# Define Graph Workflow

graph = StateGraph(EssayState)

# Nodes
graph.add_node("clarity_analysis", clarity_analysis)
graph.add_node("depth_analysis", depth_analysis)
graph.add_node("grammar_analysis", grammar_analysis)
graph.add_node("summary", summary)

# Edges (parallel start)
graph.add_edge(START, "clarity_analysis")
graph.add_edge(START, "depth_analysis")
graph.add_edge(START, "grammar_analysis")

# Join at summary
graph.add_edge("clarity_analysis", "summary")
graph.add_edge("depth_analysis", "summary")
graph.add_edge("grammar_analysis", "summary")

graph.add_edge("summary", END)

# Compile the workflow
workflow = graph.compile()


initial_state = """Machine Learning: Revolutionizing the Digital Age
Introduction
In the modern era of technological advancement, machine learning (ML) stands as one of the most transformative fields in computer science. It has revolutionized how we analyze data, make decisions, and interact with technology. Machine learning, a subset of artificial intelligence (AI), enables computers to learn patterns and make predictions without being explicitly programmed. From recommendation systems on Netflix and YouTube to advanced medical diagnostics and autonomous vehicles, machine learning is now deeply integrated into nearly every aspect of human life. Its growing influence raises profound implications for society, economics, ethics, and the future of human-computer interaction.
Definition and Core Concept
At its core, machine learning refers to the process by which computer systems improve their performance on a task through experience. Arthur Samuel, one of the pioneers in the field, defined machine learning in 1959 as “the field of study that gives computers the ability to learn without being explicitly programmed.” The key idea is that a machine can analyze large amounts of data, identify patterns, and make informed decisions or predictions based on that information.
Machine learning models are typically trained using data—known as the “training set”—which helps them identify correlations or patterns. The accuracy of these models depends largely on the quantity and quality of the data provided. Algorithms such as linear regression, decision trees, support vector machines (SVM), and neural networks allow machines to recognize complex relationships that may be difficult or impossible for humans to detect.
Historical Development of Machine Learning
The roots of machine learning date back to the mid-20th century. Alan Turing’s question, “Can machines think?” in 1950, sparked interest in artificial intelligence research. Early developments focused on symbolic reasoning and rule-based systems. In the 1950s and 1960s, algorithms like the perceptron laid the groundwork for neural networks. However, limited computational power and data scarcity slowed progress.
The 1980s saw the emergence of the “AI winter,” a period of reduced funding and interest due to unmet expectations. Yet, in the 1990s and early 2000s, the field reemerged with the rise of data-driven approaches and statistical learning. The advent of large-scale computing and the Internet fueled this resurgence. Breakthroughs in deep learning during the 2010s, particularly with convolutional neural networks (CNNs) and recurrent neural networks (RNNs), led to dramatic improvements in image recognition, speech processing, and natural language understanding. Today, with the integration of big data and cloud computing, machine learning has become central to both academic research and industrial innovation.
Types of Machine Learning
Machine learning can be broadly categorized into three main types: supervised, unsupervised, and reinforcement learning.
1. Supervised Learning:
This is the most common type of machine learning. In supervised learning, the algorithm is trained on a labeled dataset, meaning that each input data point is paired with a known output. The goal is to learn a mapping from inputs to outputs so that the model can predict outcomes for new data. Examples include spam email detection, credit risk assessment, and medical image classification.
2. Unsupervised Learning:
In unsupervised learning, the algorithm is given data without labels. It seeks to identify hidden patterns or structures within the dataset. Techniques such as clustering (e.g., K-means) and dimensionality reduction (e.g., Principal Component Analysis) fall into this category. Applications include customer segmentation in marketing and anomaly detection in cybersecurity.
3. Reinforcement Learning:
Reinforcement learning (RL) involves an agent that learns to make decisions through trial and error in an interactive environment. It receives rewards or penalties based on its actions and adjusts its strategy accordingly. RL has achieved remarkable success in game-playing AI (such as AlphaGo), robotics, and autonomous navigation.
Applications of Machine Learning
The practical applications of machine learning are vast and continue to expand across diverse sectors.
1. Healthcare:
In medicine, ML algorithms assist in disease diagnosis, personalized treatment, and drug discovery. For instance, deep learning models can analyze medical images to detect tumors or fractures with accuracy comparable to human radiologists. Predictive analytics also help identify patients at risk of chronic diseases such as diabetes or heart failure.
2. Finance:
Machine learning has transformed financial services through applications like fraud detection, algorithmic trading, and credit scoring. ML models can detect suspicious transactions in real time and predict market trends using historical data.
3. Transportation and Autonomous Systems:
Self-driving cars rely heavily on ML algorithms to process sensory data, recognize obstacles, and make split-second driving decisions. Similarly, ML helps optimize traffic flow, reduce fuel consumption, and improve logistics efficiency.
4. Education:
Machine learning enhances personalized learning experiences. Intelligent tutoring systems analyze student performance data to tailor content according to individual strengths and weaknesses. Additionally, ML assists in plagiarism detection and automated grading.
5. Marketing and E-commerce:
Recommendation systems used by Amazon, Netflix, and Spotify are prime examples of ML in marketing. These systems analyze user behavior to suggest products, shows, or music that align with user preferences, thereby improving engagement and sales.
6. Cybersecurity:
ML models can detect and prevent cyberattacks by identifying abnormal network activity or malicious software behavior. Intrusion detection systems, spam filters, and phishing detection tools rely on continuous learning from new data to improve their accuracy.
Challenges in Machine Learning
Despite its remarkable potential, machine learning faces several technical, ethical, and societal challenges.
1. Data Quality and Bias:
The performance of an ML model depends on the quality of the data it is trained on. Biased or unrepresentative data can lead to unfair or discriminatory outcomes. For example, facial recognition systems have been criticized for showing higher error rates for certain demographic groups due to unbalanced training datasets.
2. Interpretability and Transparency:
Many ML models, especially deep neural networks, are often described as “black boxes” because their internal decision-making processes are difficult to interpret. This lack of transparency poses issues in high-stakes areas like healthcare or law enforcement, where accountability is crucial.
3. Overfitting and Generalization:
Overfitting occurs when a model learns the training data too well, capturing noise rather than general patterns. Such models perform poorly on unseen data. Achieving a balance between accuracy and generalization remains a major challenge.
4. Computational Cost:
Training complex ML models requires immense computational power and energy consumption. Large-scale deep learning models often demand high-end GPUs and extended training times, raising concerns about sustainability.
5. Ethical and Privacy Concerns:
The use of personal data in ML applications raises privacy issues. There are also ethical concerns regarding surveillance, manipulation of online behavior, and job displacement due to automation.
Recent Advances and Trends
Recent developments in machine learning are pushing the boundaries of what machines can achieve. Deep learning, inspired by the human brain’s neural networks, continues to dominate the field, enabling breakthroughs in natural language processing (NLP) and computer vision. Transformer-based models such as GPT, BERT, and CLIP have revolutionized how machines understand language and images.
Another growing area is federated learning, which allows multiple devices to collaboratively train models without sharing raw data, enhancing privacy and security. Explainable AI (XAI) is also gaining traction, focusing on making machine learning models more transparent and interpretable to humans. Moreover, quantum machine learning—an intersection of quantum computing and ML—holds promise for solving complex problems much faster than classical methods.
The Future of Machine Learning
The future of machine learning is promising yet uncertain. As algorithms become more sophisticated, their integration into everyday life will only deepen. In the coming decades, ML will likely revolutionize industries such as healthcare, energy, agriculture, and climate science. It will enable predictive maintenance in infrastructure, smarter resource allocation, and personalized digital experiences.
However, with great power comes great responsibility. Policymakers, engineers, and ethicists must work together to establish frameworks that ensure transparency, fairness, and accountability in machine learning systems. Equitable access to ML technologies will also be vital to prevent widening economic and social inequalities between technologically advanced and developing nations.
Conclusion
Machine learning represents one of the most powerful technological advancements of the 21st century. Its ability to extract insights from vast datasets and make autonomous decisions is reshaping industries and redefining human-computer interaction. Yet, as it evolves, it brings new challenges related to ethics, transparency, and governance. To harness its full potential, humanity must balance innovation with responsibility. The ultimate goal should not be to replace human intelligence but to enhance it—creating a future where humans and machines work together to solve the world’s most complex problems.
"""


result = workflow.invoke(initial_state)
print(result)
