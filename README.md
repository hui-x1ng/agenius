# AI Math Tutoring System

An intelligent mathematics tutoring platform powered by AWS Bedrock and Knowledge Base technology, providing personalized learning experiences for Calculus and Linear Algebra students.

## 🚀 Features

- **Interactive Question Practice**: Random question generation from curated question banks
- **AI-Powered Tutoring**: Contextual explanations and step-by-step guidance
- **Multi-Subject Support**: Calculus and Linear Algebra modules
- **Real-time Assessment**: Instant answer checking with normalized comparison
- **Progress Tracking**: Session-based statistics and accuracy metrics
- **Knowledge Base Integration**: RAG-enhanced responses for comprehensive tutoring

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.x
- **AI/ML**: AWS Bedrock (Amazon Nova Micro v1:0)
- **Knowledge Base**: AWS Bedrock Knowledge Base
- **Storage**: Amazon S3
- **Configuration**: Environment variables with dotenv

## 📋 Prerequisites

- Python 3.8+
- AWS Account with appropriate permissions
- AWS CLI configured or IAM credentials
- S3 bucket with question data
- Bedrock Knowledge Bases (optional but recommended)

## ⚡ Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-math-tutoring-system
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=amazon.nova-micro-v1:0
S3_BUCKET=your-s3-bucket-name
S3_KEY_CALCULUS=calculus-questions.json
S3_KEY_LINEAR_ALGEBRA=linear-algebra-questions.json
KNOWLEDGE_BASE_ID_CALCULUS=your-calculus-kb-id
KNOWLEDGE_BASE_ID_LINEAR_ALGEBRA=your-linear-algebra-kb-id
GEN_MAX_TOKENS=800
GEN_TEMPERATURE=0.1
GEN_TOP_P=0.9
```

### 3. AWS Setup

#### S3 Question Banks
Upload your question files to S3 in the following JSON format:

```json
{
  "questions": [
    {
      "id": "calc_001",
      "question": "Find the derivative of f(x) = x³ + 2x² - 5x + 3",
      "standard_solution": {
        "answer": "3x² + 4x - 5",
        "steps": [
          "Apply power rule to each term",
          "d/dx(x³) = 3x²",
          "d/dx(2x²) = 4x",
          "d/dx(-5x) = -5",
          "d/dx(3) = 0"
        ]
      },
      "difficulty_level": "intermediate",
      "estimated_time_minutes": 3
    }
  ]
}
```

#### IAM Permissions
Ensure your AWS credentials have the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/amazon.nova-micro-v1:0"
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock-agent:RetrieveAndGenerate"
      ],
      "Resource": "arn:aws:bedrock:*:*:knowledge-base/*"
    }
  ]
}
```

### 4. Run the Application

```bash
streamlit run main.py
```

Navigate to `http://localhost:8501` to access the application.

## 📁 Project Structure

```
ai-math-tutoring-system/
├── main.py                 # Main application file
├── .env                    # Environment configuration
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── ARCHITECTURE.md        # Technical architecture document
└── data/                  # Sample question data (optional)
    ├── calculus_questions.json
    └── linear_algebra_questions.json
```

## 🎯 Usage

### Getting Started
1. **Select Topic**: Choose between Calculus and Linear Algebra from the sidebar
2. **Generate Question**: Click "Get New Question" to load a random practice question
3. **Submit Answer**: Enter your solution and click "Submit Answer" for instant feedback
4. **Ask AI Tutor**: Use the chat interface for additional help and explanations

### Features Overview

#### Question Practice
- Random question selection from topic-specific pools
- Multiple difficulty levels
- Time estimation for each question
- Immediate answer validation

#### AI Tutoring Chat
- Context-aware responses based on current question
- Knowledge base integration for comprehensive explanations
- Step-by-step problem-solving guidance
- Mathematical concept clarification

#### Progress Tracking
- Session statistics (questions attempted, correct answers)
- Accuracy percentage calculation
- Chat message history
- Reset functionality for fresh sessions

## 🔧 Configuration Options

### Model Parameters
- **Temperature (0.1)**: Controls response randomness
- **Top-P (0.9)**: Nucleus sampling parameter
- **Max Tokens (800)**: Maximum response length

### Question Difficulty Levels
- `beginner`: Basic concepts and simple calculations
- `intermediate`: Standard problem-solving (default)
- `advanced`: Complex multi-step problems

## 🚨 Troubleshooting

### Common Issues

**AWS Credentials Error**
```
Solution: Ensure AWS credentials are properly configured
- Check .env file
- Verify IAM permissions
- Test AWS CLI access
```

**Question Loading Failed**
```
Solution: Verify S3 configuration
- Check bucket name and keys
- Validate JSON format
- Ensure read permissions
```

**Bedrock Model Access**
```
Solution: Check model availability
- Verify region support
- Request model access if needed
- Check model ID format
```

### Debug Mode
Add debug logging by setting environment variable:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
```

## 📈 Performance & Optimization

### System Performance Tips
- **Question Caching**: Questions load per request; implement Redis for production
- **Session Management**: Optimize Streamlit session state for better UX
- **AWS Cost Control**: Monitor Bedrock token usage and implement limits
- **Response Speed**: Adjust temperature/max_tokens for faster AI responses

### Recommended Production Setup
```yaml
# docker-compose.yml example
version: '3.8'
services:
  agenius:
    build: .
    ports:
      - "8501:8501"
    environment:
      - AWS_REGION=us-east-1
    volumes:
      - ./.env:/app/.env
```

## 🚀 Future Enhancements

### Planned Features (Roadmap)
- [ ] **User Authentication**: Multi-user support with individual progress
- [ ] **Advanced Analytics**: Learning curve analysis and recommendations  
- [ ] **Mobile App**: React Native companion application
- [ ] **Collaboration**: Study groups and peer learning features
- [ ] **Content Management**: Admin dashboard for question management
- [ ] **Multi-language**: Support for Chinese, Spanish, and French
- [ ] **Integration**: LMS compatibility (Canvas, Blackboard, Moodle)

### Technical Improvements
- [ ] **Database Migration**: Replace session state with PostgreSQL
- [ ] **Microservices**: Containerized architecture with Docker
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **Monitoring**: Application performance monitoring (APM)
- [ ] **Security**: OAuth integration and data encryption

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🛠️ Development Guidelines
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 📝 Code Standards
- Follow **PEP 8** style guidelines
- Add **type hints** for all functions
- Include **docstrings** for complex functions
- Write **unit tests** for new features
- Handle **exceptions** gracefully

### 🧪 Testing
```bash
# Run tests (when available)
pytest tests/

# Code formatting
black main.py

# Linting
flake8 main.py
```

### 📊 Areas for Contribution
- **Question Banks**: Add more high-quality math problems
- **UI/UX**: Improve interface design and user experience
- **Documentation**: Enhance guides and tutorials
- **Performance**: Optimize application speed and efficiency
- **Features**: Implement items from the roadmap

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- Streamlit: Apache License 2.0
- AWS SDK: Apache License 2.0
- Python: Python Software Foundation License

## 🙏 Acknowledgments

### Special Thanks
- **AWS Bedrock Team** for providing excellent foundation models
- **Streamlit Community** for the fantastic web framework
- **Mathematical Community** for question inspiration and validation
- **Open Source Contributors** who helped improve this project

### Research & Inspiration
- *Mathematical pedagogy research* from Stanford and MIT
- *AI tutoring systems* academic papers and implementations
- *Educational technology* best practices and user experience studies

---

<div align="center">

**Built with ❤️ by [hui-x1ng](https://github.com/hui-x1ng)**

*Empowering mathematical learning through AI technology*

[![GitHub stars](https://img.shields.io/github/stars/hui-x1ng/agenius?style=social)](https://github.com/hui-x1ng/agenius/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/hui-x1ng/agenius?style=social)](https://github.com/hui-x1ng/agenius/network/members)
[![GitHub issues](https://img.shields.io/github/issues/hui-x1ng/agenius)](https://github.com/hui-x1ng/agenius/issues)

[⭐ Star this repo](https://github.com/hui-x1ng/agenius) | [🐛 Report Bug](https://github.com/hui-x1ng/agenius/issues) | [💡 Request Feature](https://github.com/hui-x1ng/agenius/issues)

</div>