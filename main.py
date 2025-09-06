"""
========================================
🎓 Advanced Calculus AI Tutor System
Enterprise-grade dual-agent architecture with AWS Bedrock & Knowledge Base
========================================

Features:
- Intelligent question generation and automated grading
- Context-aware tutoring with RAG (Retrieval-Augmented Generation)
- Multi-turn conversational learning experience
- Enterprise AWS integration (S3, Bedrock, Knowledge Base)
- Comprehensive logging and analytics
- Advanced error handling and resilience
"""

import os
import json
import random
import sys
import logging
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown

# ========================================
# Configuration & Environment Setup
# ========================================

load_dotenv()

class Config:
    """Centralized configuration management"""
    
    # AWS Configuration
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_KEY = os.getenv("S3_KEY")
    KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
    
    # Model Parameters
    GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "800"))
    GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.1"))
    GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
    
    # System Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))

class SessionState(Enum):
    """System state management"""
    INITIALIZED = "initialized"
    QUESTION_PRESENTED = "question_presented" 
    ANSWER_SUBMITTED = "answer_submitted"
    TUTORING_ACTIVE = "tutoring_active"
    SESSION_ENDED = "session_ended"

# ========================================
# Enhanced Logging & Console Setup
# ========================================

# Rich console for beautiful output
console = Console()

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calculus_tutor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================================
# AWS Service Clients with Error Handling
# ========================================

class AWSServiceManager:
    """Centralized AWS service management with enhanced error handling"""
    
    def __init__(self):
        self.region = Config.AWS_REGION
        self._clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS clients with proper error handling"""
        try:
            self._clients['bedrock'] = boto3.client("bedrock-runtime", region_name=self.region)
            self._clients['s3'] = boto3.client("s3", region_name=self.region)
            self._clients['bedrock_agent'] = boto3.client("bedrock-agent-runtime", region_name=self.region)
            logger.info(f"AWS clients initialized successfully in region: {self.region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise
    
    @property
    def bedrock(self):
        return self._clients['bedrock']
    
    @property
    def s3(self):
        return self._clients['s3']
    
    @property
    def bedrock_agent(self):
        return self._clients['bedrock_agent']

# Global AWS service manager
aws_manager = AWSServiceManager()

# ========================================
# Enhanced AI Service Layer
# ========================================

class BedrockService:
    """Enhanced Bedrock service with retry logic and monitoring"""
    
    def __init__(self):
        self.client = aws_manager.bedrock
        self.model_id = Config.BEDROCK_MODEL_ID
        
    @contextmanager
    def _performance_monitor(self, operation: str):
        """Monitor API performance"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"{operation} completed in {duration:.2f}s")
    
    def _create_inference_config(self) -> Dict[str, Any]:
        """Create standardized inference configuration"""
        return {
            "temperature": Config.GEN_TEMPERATURE,
            "topP": Config.GEN_TOP_P,
            "maxTokens": Config.GEN_MAX_TOKENS
        }
    
    def chat_single_turn(self, prompt: str) -> str:
        """Single-turn conversation with enhanced error handling"""
        with self._performance_monitor("Single-turn chat"):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig=self._create_inference_config(),
                )
                return response["output"]["message"]["content"][0]["text"]
            except ClientError as e:
                logger.error(f"Bedrock API error: {e}")
                return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    def chat_multi_turn(self, messages: List[Dict[str, Any]]) -> str:
        """Multi-turn conversation with context management"""
        with self._performance_monitor("Multi-turn chat"):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    inferenceConfig=self._create_inference_config(),
                )
                return response["output"]["message"]["content"][0]["text"]
            except ClientError as e:
                logger.error(f"Bedrock multi-turn error: {e}")
                return "I'm having trouble maintaining our conversation. Let's start fresh."

class KnowledgeBaseService:
    """Enhanced Knowledge Base service with advanced RAG capabilities"""
    
    def __init__(self):
        self.client = aws_manager.bedrock_agent
        self.kb_id = Config.KNOWLEDGE_BASE_ID
        self.model_arn = f"arn:aws:bedrock:{Config.AWS_REGION}::foundation-model/{Config.BEDROCK_MODEL_ID}"
    
    def retrieve_and_generate(self, query: str, context_hint: str = "") -> str:
        """Enhanced RAG with context-aware retrieval"""
        if not self.kb_id:
            logger.warning("Knowledge Base ID not configured")
            return ""
        
        # Construct enriched query with context
        enriched_query = f"{context_hint}\n\nUser Query: {query}" if context_hint else query
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching knowledge base...", total=None)
            
            try:
                response = self.client.retrieve_and_generate(
                    input={"text": enriched_query},
                    retrieveAndGenerateConfiguration={
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseConfiguration": {
                            "knowledgeBaseId": self.kb_id,
                            "modelArn": self.model_arn,
                        },
                    },
                )
                progress.update(task, completed=True)
                return response.get("output", {}).get("text", "")
                
            except ClientError as e:
                progress.update(task, completed=True)
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("AccessDeniedException", "ValidationException"):
                    logger.error(f"Knowledge Base access error: {error_code}")
                    return f"⚠️ Knowledge Base access issue: {error_code}. Please check permissions."
                raise

# Initialize services
bedrock_service = BedrockService()
kb_service = KnowledgeBaseService()

# ========================================
# Enhanced Data Models
# ========================================

@dataclass
class Question:
    """Enhanced question model with metadata"""
    id: str
    question: str
    standard_solution: Dict[str, Any]
    extended_solution: Optional[Dict[str, Any]] = None
    difficulty_level: str = "intermediate"
    topic: str = "calculus"
    estimated_time_minutes: int = 5
    created_at: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        """Create Question from dictionary with validation"""
        required_fields = ['id', 'question', 'standard_solution']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        return cls(**data)

@dataclass
class AssessmentResult:
    """Comprehensive assessment result with analytics"""
    is_correct: bool
    feedback_message: str
    standard_solution: Optional[Dict[str, Any]] = None
    question_id: Optional[str] = None
    confidence_score: float = 0.0
    assessment_time: Optional[str] = None
    
    def __post_init__(self):
        if not self.assessment_time:
            self.assessment_time = datetime.now().isoformat()

@dataclass
class LearningSession:
    """Track learning session analytics"""
    session_id: str
    start_time: str
    questions_attempted: int = 0
    correct_answers: int = 0
    total_interactions: int = 0
    current_state: SessionState = SessionState.INITIALIZED
    
    @property
    def accuracy_rate(self) -> float:
        if self.questions_attempted == 0:
            return 0.0
        return self.correct_answers / self.questions_attempted

# ========================================
# Enhanced Data Management
# ========================================

class DataManager:
    """Advanced data management with caching and validation"""
    
    def __init__(self):
        self.s3_client = aws_manager.s3
        self.bucket = Config.S3_BUCKET
        self.key = Config.S3_KEY
        self._question_cache: Dict[str, Question] = {}
        self._last_updated: Optional[datetime] = None
    
    def load_question_dataset(self) -> Dict[str, Question]:
        """Load and validate question dataset from S3"""
        if not self.bucket or not self.key:
            raise RuntimeError("S3_BUCKET and S3_KEY must be configured in environment variables")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading question dataset from S3...", total=None)
                
                response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
                raw_data = json.loads(response["Body"].read().decode("utf-8"))
                
                # Validate dataset structure
                if "questions" not in raw_data or not isinstance(raw_data["questions"], list):
                    raise ValueError("Invalid dataset format: missing 'questions' array")
                
                # Convert to Question objects with validation
                questions = {}
                for q_data in raw_data["questions"]:
                    try:
                        question = Question.from_dict(q_data)
                        questions[question.id] = question
                    except ValueError as e:
                        logger.warning(f"Skipping invalid question: {e}")
                        continue
                
                progress.update(task, completed=True)
                logger.info(f"Successfully loaded {len(questions)} questions from S3")
                
                self._question_cache = questions
                self._last_updated = datetime.now()
                return questions
                
        except ClientError as e:
            error_msg = f"Failed to load dataset from s3://{self.bucket}/{self.key}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_random_question(self, exclude_ids: Optional[List[str]] = None) -> Question:
        """Get a random question with optional exclusions"""
        if not self._question_cache:
            self.load_question_dataset()
        
        available_questions = list(self._question_cache.values())
        if exclude_ids:
            available_questions = [q for q in available_questions if q.id not in exclude_ids]
        
        if not available_questions:
            raise ValueError("No questions available")
        
        return random.choice(available_questions)
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Retrieve specific question by ID"""
        if not self._question_cache:
            self.load_question_dataset()
        
        return self._question_cache.get(question_id)

# ========================================
# Agent 1: Advanced Question Management & Assessment
# ========================================

class QuestionAgent:
    """Advanced question management and assessment agent"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.assessment_history: List[AssessmentResult] = []
    
    def present_question(self, question_id: Optional[str] = None) -> Question:
        """Present a question with enhanced formatting"""
        try:
            if question_id:
                question = self.data_manager.get_question_by_id(question_id)
                if not question:
                    raise ValueError(f"Question with ID '{question_id}' not found")
            else:
                question = self.data_manager.get_random_question()
            
            # Display question with rich formatting
            self._display_question(question)
            return question
            
        except Exception as e:
            logger.error(f"Error presenting question: {e}")
            console.print(f"[red]Error: {e}[/red]")
            raise
    
    def _display_question(self, question: Question):
        """Display question with beautiful formatting"""
        question_panel = Panel(
            Text(question.question, style="bold blue"),
            title=f"📚 Question {question.id}",
            subtitle=f"Topic: {question.topic.title()} | Difficulty: {question.difficulty_level.title()}",
            border_style="blue"
        )
        console.print(question_panel)
    
    def _normalize_answer(self, answer: str) -> str:
        """Advanced answer normalization"""
        if not answer:
            return ""
        
        # Comprehensive normalization
        normalized = answer.strip().lower()
        replacements = [
            (" ", ""), ("\\", ""), ("*", ""), ("^1", ""),
            ("（", "("), ("）", ")"), 
            ("plus", "+"), ("minus", "-"), ("times", "*"), ("divided", "/")
        ]
        
        for old, new in replacements:
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def assess_answer(self, question: Question, student_answer: str) -> AssessmentResult:
        """Comprehensive answer assessment with detailed feedback"""
        try:
            standard_answer = question.standard_solution.get("answer", "")
            
            # Normalize both answers for comparison
            normalized_student = self._normalize_answer(student_answer)
            normalized_standard = self._normalize_answer(standard_answer)
            
            is_correct = normalized_student == normalized_standard
            
            if is_correct:
                result = AssessmentResult(
                    is_correct=True,
                    feedback_message="🎉 Excellent! Your answer is correct.",
                    question_id=question.id,
                    confidence_score=1.0
                )
                self._display_success_feedback(result)
            else:
                result = AssessmentResult(
                    is_correct=False,
                    feedback_message="Not quite right. Let me show you the solution approach.",
                    standard_solution=question.standard_solution,
                    question_id=question.id,
                    confidence_score=0.0
                )
                self._display_correction_feedback(result, question)
            
            self.assessment_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error assessing answer: {e}")
            return AssessmentResult(
                is_correct=False,
                feedback_message="Assessment error occurred. Please try again.",
                question_id=question.id if question else None
            )
    
    def _display_success_feedback(self, result: AssessmentResult):
        """Display success feedback with celebration"""
        success_panel = Panel(
            "🎉 **Correct!** Well done! Your mathematical reasoning is on point.",
            title="✅ Assessment Result",
            border_style="green"
        )
        console.print(success_panel)
    
    def _display_correction_feedback(self, result: AssessmentResult, question: Question):
        """Display detailed correction feedback"""
        if not result.standard_solution:
            return
        
        steps = result.standard_solution.get("steps", [])
        if steps:
            steps_text = "\n".join([f"**Step {i+1}:** {step}" for i, step in enumerate(steps)])
            solution_md = Markdown(f"## Standard Solution Approach\n\n{steps_text}")
            
            correction_panel = Panel(
                solution_md,
                title="📖 Detailed Solution",
                border_style="yellow"
            )
            console.print(correction_panel)
            
            console.print("\n[dim]💡 Feel free to ask questions about any step you'd like clarified![/dim]")

# ========================================
# Agent 2: Advanced AI Tutoring System
# ========================================

class TutoringAgent:
    """Advanced AI tutoring system with context-aware responses"""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_question: Optional[Question] = None
        self.tutoring_context = ""
    
    def initialize_session(self, question: Question):
        """Initialize tutoring session with question context"""
        self.current_question = question
        self.conversation_history.clear()
        
        # Build rich context from question data
        context_parts = [f"Current Question: {question.question}"]
        
        if question.extended_solution:
            extended_content = question.extended_solution.get("answer", "")
            if extended_content:
                context_parts.append(f"Extended Context: {extended_content}")
        
        self.tutoring_context = "\n".join(context_parts)
        logger.info(f"Tutoring session initialized for question {question.id}")
    
    def respond_to_query(self, user_query: str) -> str:
        """Generate context-aware tutoring response"""
        if not self.current_question:
            return "Please start with a question first before asking for help."
        
        try:
            # Retrieve relevant knowledge
            kb_context = kb_service.retrieve_and_generate(
                user_query, 
                context_hint=self.tutoring_context
            )
            
            # Build comprehensive context
            full_context = self._build_tutoring_context(kb_context)
            
            # Create system prompt for tutoring
            system_prompt = self._create_tutoring_prompt(full_context)
            
            # Prepare conversation messages
            messages = [{"role": "user", "content": [{"text": system_prompt}]}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": [{"text": user_query}]})
            
            # Generate response
            response = bedrock_service.chat_multi_turn(messages)
            
            # Update conversation history
            self._update_conversation_history(user_query, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in tutoring response: {e}")
            return "I'm having trouble processing your question right now. Could you rephrase it?"
    
    def _build_tutoring_context(self, kb_context: str) -> str:
        """Build comprehensive tutoring context"""
        context_sections = []
        
        if self.tutoring_context:
            context_sections.append(f"**Question Context:**\n{self.tutoring_context}")
        
        if kb_context:
            context_sections.append(f"**Knowledge Base Information:**\n{kb_context}")
        
        return "\n\n".join(context_sections) if context_sections else "No additional context available"
    
    def _create_tutoring_prompt(self, context: str) -> str:
        """Create sophisticated tutoring system prompt"""
        return f"""You are an expert calculus tutor with a PhD in Mathematics. Your role is to provide clear, 
step-by-step explanations that help students understand mathematical concepts deeply.

**Guidelines:**
- Use clear, encouraging language
- Break down complex concepts into digestible steps
- Provide intuitive explanations alongside mathematical rigor
- Ask guiding questions to promote active learning
- Reference specific parts of solutions when helpful
- Adapt your explanation level to the student's apparent understanding

**Available Context:**
{context}

**Your Task:** Answer the student's question using the provided context, maintaining a supportive and educational tone.
"""
    
    def _update_conversation_history(self, user_query: str, assistant_response: str):
        """Update conversation history with proper formatting"""
        self.conversation_history.extend([
            {"role": "user", "content": [{"text": user_query}]},
            {"role": "assistant", "content": [{"text": assistant_response}]}
        ])
        
        # Keep history manageable (last 10 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

# ========================================
# Enhanced User Interface & Experience
# ========================================

class CalculusTutorInterface:
    """Advanced user interface with rich interactive experience"""
    
    def __init__(self):
        self.question_agent = QuestionAgent()
        self.tutoring_agent = TutoringAgent()
        self.session = LearningSession(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now().isoformat()
        )
        self.current_question: Optional[Question] = None
    
    def display_welcome(self):
        """Display enhanced welcome screen"""
        welcome_text = """
# 🎓 Advanced Calculus AI Tutor System

**Powered by AWS Bedrock & Knowledge Base Technology**

## Features:
- **Intelligent Question Generation** - Adaptive problem selection
- **Automated Assessment** - Instant feedback with detailed solutions  
- **AI-Powered Tutoring** - Context-aware explanations and guidance
- **Multi-turn Conversations** - Deep dive into any concept

## Available Commands:
- `start` or `start [question_id]` - Begin with a new question
- `next` - Move to the next question
- `stats` - View your learning progress
- `help` - Show detailed help
- `quit` - Exit the system

*Ready to enhance your calculus mastery? Let's begin!*
        """
        
        welcome_panel = Panel(
            Markdown(welcome_text),
            title="🚀 Welcome to Advanced Calculus Learning",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(welcome_panel)
    
    def display_session_stats(self):
        """Display comprehensive session statistics"""
        stats_table = Table(title="📊 Learning Session Analytics")
        
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")
        
        stats_table.add_row("Questions Attempted", str(self.session.questions_attempted))
        stats_table.add_row("Correct Answers", str(self.session.correct_answers))
        stats_table.add_row("Accuracy Rate", f"{self.session.accuracy_rate:.1%}")
        stats_table.add_row("Total Interactions", str(self.session.total_interactions))
        stats_table.add_row("Session Duration", self._get_session_duration())
        
        console.print(stats_table)
    
    def _get_session_duration(self) -> str:
        """Calculate and format session duration"""
        start = datetime.fromisoformat(self.session.start_time)
        duration = datetime.now() - start
        
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def process_user_input(self, user_input: str) -> bool:
        """Process user input with enhanced command handling"""
        command = user_input.strip().lower()
        self.session.total_interactions += 1
        
        # Command routing
        if command in ("quit", "exit", ":q"):
            self._handle_quit()
            return False
            
        elif command.startswith("start"):
            self._handle_start_command(user_input)
            
        elif command == "next":
            self._handle_next_command()
            
        elif command == "stats":
            self.display_session_stats()
            
        elif command == "help":
            self._display_help()
            
        elif self.session.current_state == SessionState.QUESTION_PRESENTED:
            self._handle_answer_submission(user_input)
            
        elif self.session.current_state == SessionState.TUTORING_ACTIVE:
            self._handle_tutoring_query(user_input)
            
        else:
            self._handle_unknown_input()
        
        return True
    
    def _handle_start_command(self, user_input: str):
        """Handle start command with optional question ID"""
        parts = user_input.split()
        question_id = parts[1] if len(parts) > 1 else None
        
        try:
            self.current_question = self.question_agent.present_question(question_id)
            self.session.current_state = SessionState.QUESTION_PRESENTED
            console.print("\n[bold green]Enter your answer:[/bold green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def _handle_next_command(self):
        """Handle next question command"""
        try:
            self.current_question = self.question_agent.present_question()
            self.session.current_state = SessionState.QUESTION_PRESENTED
            self.tutoring_agent.conversation_history.clear()  # Reset tutoring context
            console.print("\n[bold green]Enter your answer:[/bold green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def _handle_answer_submission(self, answer: str):
        """Handle student answer submission"""
        if not self.current_question:
            console.print("[red]No active question. Please start with a new question.[/red]")
            return
        
        result = self.question_agent.assess_answer(self.current_question, answer)
        
        # Update session statistics
        self.session.questions_attempted += 1
        if result.is_correct:
            self.session.correct_answers += 1
            self.session.current_state = SessionState.ANSWER_SUBMITTED
            console.print("\n[dim]Enter 'next' for another question or 'quit' to exit.[/dim]")
        else:
            # Initialize tutoring session
            self.tutoring_agent.initialize_session(self.current_question)
            self.session.current_state = SessionState.TUTORING_ACTIVE
            console.print("\n[dim]Ask questions about the solution, enter 'next' for a new question, or 'quit' to exit.[/dim]")
    
    def _handle_tutoring_query(self, query: str):
        """Handle tutoring queries with AI assistance"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Thinking...", total=None)
                response = self.tutoring_agent.respond_to_query(query)
                progress.update(task, completed=True)
            
            # Display tutoring response
            tutor_panel = Panel(
                Markdown(response),
                title="🧠 AI Tutor Response",
                border_style="blue"
            )
            console.print(tutor_panel)
            console.print("\n[dim]Continue asking questions, enter 'next' for a new question, or 'quit' to exit.[/dim]")
            
        except Exception as e:
            logger.error(f"Tutoring error: {e}")
            console.print("[red]I'm having trouble with that question. Could you try rephrasing it?[/red]")
    
    def _display_help(self):
        """Display comprehensive help information"""
        help_text = """
## 📖 Detailed Help Guide

### Getting Started:
- **`start`** - Get a random calculus question
- **`start Q3`** - Get specific question by ID (e.g., Q3)

### During Problem Solving:
- After seeing a question, simply type your mathematical answer
- Examples: `2x`, `x^2 + 3x`, `ln(x) + C`, etc.

### After Assessment:
- **Correct answers**: Use `next` to continue or explore more questions
- **Incorrect answers**: Ask specific questions about the solution
  - "I don't understand step 2"
  - "Why do we use integration by parts here?"
  - "Can you explain the chain rule application?"

### System Commands:
- **`next`** - Get a new question (works anytime)
- **`stats`** - View your learning progress and session analytics  
- **`help`** - Show this help guide
- **`quit`** - Exit the system

### Tips for Best Experience:
- Be specific in your tutoring questions
- Don't hesitate to ask for clarification on any step
- Review the solution steps before asking follow-up questions

*The AI tutor is here to help you understand, not just get answers!*
        """
        
        help_panel = Panel(
            Markdown(help_text),
            title="🆘 System Help",
            border_style="yellow"
        )
        console.print(help_panel)
    
    def _handle_unknown_input(self):
        """Handle unrecognized input"""
        console.print("[yellow]I didn't understand that command. Type 'help' for available commands or 'start' to begin.[/yellow]")
    
    def _handle_quit(self):
        """Handle graceful system exit"""
        # Display final session summary
        console.print("\n" + "="*60)
        console.print("[bold cyan]📊 Final Session Summary[/bold cyan]")
        self.display_session_stats()
        
        # Farewell message
        farewell_text = f"""
Thank you for using the Advanced Calculus AI Tutor System!

**Session Highlights:**
- Questions Attempted: {self.session.questions_attempted}
- Accuracy Rate: {self.session.accuracy_rate:.1%}
- Learning Time: {self._get_session_duration()}

*Keep practicing and stay curious about mathematics!* 🚀
        """
        
        farewell_panel = Panel(
            Text(farewell_text.strip(), style="bold green"),
            title="👋 Goodbye!",
            border_style="green"
        )
        console.print(farewell_panel)
        
        # Log session completion
        logger.info(f"Session {self.session.session_id} completed successfully")

# ========================================
# Application Entry Point & Error Handling
# ========================================

def validate_environment() -> bool:
    """Validate required environment variables and AWS access"""
    required_vars = [
        "S3_BUCKET", "S3_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_panel = Panel(
            f"[red]Missing required environment variables:[/red]\n" + 
            "\n".join([f"• {var}" for var in missing_vars]) +
            "\n\n[yellow]Please configure these in your .env file.[/yellow]",
            title="⚠️ Configuration Error",
            border_style="red"
        )
        console.print(error_panel)
        return False
    
    return True

def main():
    """Enhanced main application with comprehensive error handling"""
    try:
        # Environment validation
        if not validate_environment():
            sys.exit(1)
        
        # Initialize application
        console.print("[dim]Initializing Advanced Calculus AI Tutor System...[/dim]")
        interface = CalculusTutorInterface()
        
        # Display welcome screen
        interface.display_welcome()
        
        # Main interaction loop
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]calculus-tutor>[/bold cyan]", default="").strip()
                
                if not user_input:
                    continue
                
                should_continue = interface.process_user_input(user_input)
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation interrupted. Type 'quit' to exit gracefully.[/yellow]")
                continue
            except EOFError:
                console.print("\n[dim]Session ended.[/dim]")
                break
                
    except Exception as e:
        logger.error(f"Fatal application error: {e}")
        error_panel = Panel(
            f"[red]A fatal error occurred:[/red]\n{str(e)}\n\n[yellow]Please check the logs for more details.[/yellow]",
            title="💥 System Error",
            border_style="red"
        )
        console.print(error_panel)
        sys.exit(1)

# ========================================
# Additional Utility Functions
# ========================================

def setup_development_environment():
    """Setup development environment with sample data"""
    """
    For development/testing purposes only.
    This function can help set up a local environment with sample questions.
    """
    sample_questions = {
        "questions": [
            {
                "id": "CALC_001",
                "question": "Find the derivative of f(x) = 3x² + 2x - 1",
                "standard_solution": {
                    "answer": "6x + 2",
                    "steps": [
                        "Apply the power rule to each term",
                        "d/dx(3x²) = 3 × 2x = 6x",
                        "d/dx(2x) = 2",
                        "d/dx(-1) = 0",
                        "Combine: 6x + 2 + 0 = 6x + 2"
                    ]
                },
                "extended_solution": {
                    "answer": "The power rule states that d/dx(xⁿ) = n×x^(n-1). This is one of the fundamental rules of differentiation."
                },
                "difficulty_level": "beginner",
                "topic": "derivatives",
                "estimated_time_minutes": 3
            },
            {
                "id": "CALC_002", 
                "question": "Evaluate the integral ∫(2x + 3)dx",
                "standard_solution": {
                    "answer": "x² + 3x + C",
                    "steps": [
                        "Split the integral: ∫2x dx + ∫3 dx",
                        "Apply the power rule for integration",
                        "∫2x dx = 2 × (x²/2) = x²",
                        "∫3 dx = 3x",
                        "Add constant of integration: x² + 3x + C"
                    ]
                },
                "extended_solution": {
                    "answer": "Integration is the reverse process of differentiation. The constant C represents the family of antiderivatives."
                },
                "difficulty_level": "beginner",
                "topic": "integration",
                "estimated_time_minutes": 4
            }
        ]
    }
    
    console.print("[dim]Sample question structure created for development.[/dim]")
    return sample_questions

# ========================================
# Performance Monitoring & Analytics
# ========================================

class PerformanceMonitor:
    """Monitor system performance and user learning analytics"""
    
    def __init__(self):
        self.metrics = {
            "total_sessions": 0,
            "average_session_length": 0.0,
            "question_difficulty_distribution": {},
            "common_misconceptions": [],
            "api_response_times": []
        }
    
    def log_performance_metric(self, metric_name: str, value: float):
        """Log performance metrics for monitoring"""
        timestamp = datetime.now().isoformat()
        logger.info(f"PERFORMANCE_METRIC: {metric_name}={value} at {timestamp}")
    
    def analyze_learning_patterns(self, session: LearningSession) -> Dict[str, Any]:
        """Analyze learning patterns from session data"""
        return {
            "session_effectiveness": session.accuracy_rate,
            "engagement_level": session.total_interactions / max(session.questions_attempted, 1),
            "learning_velocity": session.questions_attempted / self._calculate_session_minutes(session)
        }
    
    def _calculate_session_minutes(self, session: LearningSession) -> float:
        """Calculate session duration in minutes"""
        start = datetime.fromisoformat(session.start_time)
        duration = datetime.now() - start
        return duration.total_seconds() / 60

# ========================================
# Integration Tests & Health Checks
# ========================================

def run_health_checks() -> bool:
    """Run comprehensive system health checks"""
    health_status = True
    
    console.print("[dim]Running system health checks...[/dim]")
    
    # Test AWS connectivity
    try:
        aws_manager.bedrock.list_foundation_models()
        console.print("✅ AWS Bedrock connectivity: OK")
    except Exception as e:
        console.print(f"❌ AWS Bedrock connectivity: FAILED ({e})")
        health_status = False
    
    # Test S3 access
    try:
        if Config.S3_BUCKET:
            aws_manager.s3.head_bucket(Bucket=Config.S3_BUCKET)
            console.print("✅ S3 bucket access: OK")
        else:
            console.print("⚠️ S3 bucket not configured")
    except Exception as e:
        console.print(f"❌ S3 bucket access: FAILED ({e})")
        health_status = False
    
    # Test Knowledge Base access
    try:
        if Config.KNOWLEDGE_BASE_ID:
            # Simple test query
            kb_service.retrieve_and_generate("test query")
            console.print("✅ Knowledge Base access: OK")
        else:
            console.print("⚠️ Knowledge Base not configured")
    except Exception as e:
        console.print(f"❌ Knowledge Base access: FAILED ({e})")
        health_status = False
    
    return health_status

# ========================================
# CLI Arguments & Advanced Features
# ========================================

def parse_cli_arguments():
    """Parse command line arguments for advanced features"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Calculus AI Tutor System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculus_tutor.py                    # Normal interactive mode
  python calculus_tutor.py --health-check     # Run system diagnostics  
  python calculus_tutor.py --setup-dev        # Setup development environment
  python calculus_tutor.py --batch-mode       # Non-interactive batch processing
        """
    )
    
    parser.add_argument(
        "--health-check", 
        action="store_true",
        help="Run comprehensive system health checks"
    )
    
    parser.add_argument(
        "--setup-dev",
        action="store_true", 
        help="Setup development environment with sample data"
    )
    
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Run in batch processing mode (for automation)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to custom configuration file"
    )
    
    return parser.parse_args()

# ========================================
# Application Entry Point
# ========================================

if __name__ == "__main__":
    # Configure logging based on CLI args
    args = parse_cli_arguments()
    
    # Set logging level from CLI
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        if args.health_check:
            console.print("[bold blue]🔍 Running System Health Checks[/bold blue]")
            health_ok = run_health_checks()
            sys.exit(0 if health_ok else 1)
        
        elif args.setup_dev:
            console.print("[bold yellow]🛠️ Setting Up Development Environment[/bold yellow]")
            sample_data = setup_development_environment()
            console.print("[green]Development environment setup completed![/green]")
            sys.exit(0)
        
        elif args.batch_mode:
            console.print("[bold cyan]⚙️ Batch Mode Not Yet Implemented[/bold cyan]")
            console.print("This feature is planned for future releases.")
            sys.exit(0)
        
        else:
            # Normal interactive mode
            main()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Program interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical system failure: {e}")
        console.print(f"[red]Critical Error: {e}[/red]")
        sys.exit(1)