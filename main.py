import streamlit as st
import os
import json
import random
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import boto3
from dotenv import load_dotenv

load_dotenv()

class Config:
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_KEY_CALCULUS = os.getenv("S3_KEY_CALCULUS") 
    S3_KEY_LINEAR_ALGEBRA = os.getenv("S3_KEY_LINEAR_ALGEBRA")
    KNOWLEDGE_BASE_ID_CALCULUS = os.getenv("KNOWLEDGE_BASE_ID_CALCULUS") 
    KNOWLEDGE_BASE_ID_LINEAR_ALGEBRA = os.getenv("KNOWLEDGE_BASE_ID_LINEAR_ALGEBRA")
    GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "800"))
    GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.1"))
    GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))

@dataclass
class Question:
    id: str
    question: str
    standard_solution: Dict[str, Any]
    extended_solution: Optional[Dict[str, Any]] = None
    difficulty_level: str = "intermediate"
    estimated_time_minutes: int = 5
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        return cls(**data)

def init_app():
    if 'questions_attempted' not in st.session_state:
        st.session_state.questions_attempted = 0
    if 'correct_answers' not in st.session_state:
        st.session_state.correct_answers = 0
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'selected_topic' not in st.session_state:
        st.session_state.selected_topic = 'calculus'

def load_questions(topic: str) -> List[Dict[str, Any]]:
    try:
        s3_client = boto3.client("s3", region_name=Config.AWS_REGION)
        key = Config.S3_KEY_CALCULUS if topic == 'calculus' else Config.S3_KEY_LINEAR_ALGEBRA
        response = s3_client.get_object(Bucket=Config.S3_BUCKET, Key=key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        return data.get("questions", [])
    except Exception as e:
        st.error(f"Failed to load questions: {e}")
        return []

def get_random_question(topic: str) -> Optional[Question]:
    questions_data = load_questions(topic)
    if not questions_data:
        return None
    
    try:
        question_data = random.choice(questions_data)
        return Question.from_dict(question_data)
    except Exception:
        return None

def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    normalized = answer.strip().lower()
    replacements = [(" ", ""), ("\\", ""), ("*", ""), ("^1", "")]
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    return normalized

def check_answer(question: Question, student_answer: str) -> bool:
    standard_answer = question.standard_solution.get("answer", "")
    return normalize_answer(student_answer) == normalize_answer(standard_answer)

def get_ai_tutor_response(user_query: str, topic: str, current_question: Optional[Question] = None) -> str:
    try:
        bedrock = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)
        bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=Config.AWS_REGION)
        
        context = ""
        if current_question:
            context = f"Current Question: {current_question.question}\n"
            if current_question.extended_solution:
                context += f"Extended Info: {current_question.extended_solution.get('answer', '')}\n"
        
        kb_id = Config.KNOWLEDGE_BASE_ID_CALCULUS if topic == 'calculus' else Config.KNOWLEDGE_BASE_ID_LINEAR_ALGEBRA
        kb_context = ""
        
        if kb_id:
            try:
                kb_response = bedrock_agent.retrieve_and_generate(
                    input={"text": f"{context}User Query: {user_query}"},
                    retrieveAndGenerateConfiguration={
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseConfiguration": {
                            "knowledgeBaseId": kb_id,
                            "modelArn": f"arn:aws:bedrock:{Config.AWS_REGION}::foundation-model/{Config.BEDROCK_MODEL_ID}",
                        },
                    },
                )
                kb_context = kb_response.get("output", {}).get("text", "")
            except:
                pass
        
        prompt = f"""You are an expert mathematics tutor. Answer the student's question clearly and helpfully.

{f"Question Context: {context}" if context else ""}
{f"Knowledge Base Info: {kb_context}" if kb_context else ""}

Student Question: {user_query}

Provide a clear, step-by-step explanation that helps the student understand the concept."""

        response = bedrock.converse(
            modelId=Config.BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "temperature": Config.GEN_TEMPERATURE,
                "topP": Config.GEN_TOP_P,
                "maxTokens": Config.GEN_MAX_TOKENS
            }
        )
        
        return response["output"]["message"]["content"][0]["text"]
        
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="AI Math Tutoring System", page_icon="📚", layout="wide")
    
    init_app()
    
    st.title("📚 AI Math Tutoring System")
    st.markdown("**Powered by AWS Bedrock and Knowledge Base Technology**")
    
    with st.sidebar:
        st.header("Configuration")
        
        topic_options = {'calculus': 'Calculus', 'linear_algebra': 'Linear Algebra'}
        selected_topic = st.selectbox(
            "Select Topic",
            options=list(topic_options.keys()),
            format_func=lambda x: topic_options[x],
            index=0
        )
        st.session_state.selected_topic = selected_topic
        
        st.divider()
        st.header("Statistics")
        accuracy = st.session_state.correct_answers / max(st.session_state.questions_attempted, 1) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.questions_attempted)
            st.metric("Correct", st.session_state.correct_answers)
        with col2:
            st.metric("Accuracy", f"{accuracy:.1f}%")
            st.metric("Chat Messages", len(st.session_state.chat_messages))
        
        if st.button("Reset Session"):
            st.session_state.questions_attempted = 0
            st.session_state.correct_answers = 0
            st.session_state.current_question = None
            st.session_state.chat_messages = []
            st.session_state.show_solution = False
            st.session_state.last_answer_result = None
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📚 Question Area")
        
        if st.button("Get New Question", type="primary"):
            question = get_random_question(st.session_state.selected_topic)
            if question:
                st.session_state.current_question = question
                st.success("New question loaded!")
            else:
                st.error("Could not load question. Check configuration.")
        
        if st.session_state.current_question:
            q = st.session_state.current_question
            
            st.subheader(f"Question {q.id}")
            st.info(f"**Difficulty:** {q.difficulty_level} | **Time:** {q.estimated_time_minutes} min")
            st.markdown(f"### {q.question}")
            
            answer = st.text_input("Your answer:", placeholder="Enter your mathematical answer...")
            
            if st.button("Submit Answer"):
                if answer.strip():
                    st.session_state.questions_attempted += 1
                    is_correct = check_answer(q, answer)
                    
                    if is_correct:
                        st.session_state.correct_answers += 1
                        st.success("Correct! Well done!")
                        st.balloons()
                    else:
                        st.error("Incorrect. Here's the solution:")
                        
                        solution = q.standard_solution
                        st.subheader("Solution")
                        st.code(solution.get("answer", ""))
                        
                        steps = solution.get("steps", [])
                        if steps:
                            st.subheader("Steps")
                            for i, step in enumerate(steps, 1):
                                st.write(f"**{i}.** {step}")
                else:
                    st.warning("Please enter an answer.")
        else:
            st.info("Click 'Get New Question' to start!")
    
    with col2:
        st.header("💬 AI Tutor Chat")
        
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        if prompt := st.chat_input("Ask the AI tutor about math or the current question..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.spinner("AI is thinking..."):
                response = get_ai_tutor_response(
                    prompt, 
                    st.session_state.selected_topic, 
                    st.session_state.current_question
                )
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Clear Chat"):
            st.session_state.chat_messages = []
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Features**\n- Question generation\n- Answer assessment\n- AI tutoring")
    with col2:
        st.info("**Technology**\n- AWS Bedrock\n- Knowledge base\n- Cloud storage")
    with col3:
        st.info("**Analytics**\n- Progress tracking\n- Accuracy stats\n- Chat history")

if __name__ == "__main__":
    main()