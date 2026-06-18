# 🎓 MentorAI

### AI Classroom Intelligence System

> **AI that assists teachers, not replaces them.**

MentorAI is a voice-enabled AI Classroom Intelligence System designed to support teachers during live classroom sessions.

Unlike traditional AI chatbots that simply answer questions, MentorAI actively helps teachers explain concepts, generate quizzes, simplify difficult terms, create engaging stories, and verify student understanding through Socratic questioning.

The platform is designed specifically for classroom environments where teachers need real-time assistance without interrupting the teaching process.

---

# 🌟 Vision

Most AI systems focus on providing answers.

MentorAI focuses on improving learning.

Our vision is to transform AI from an information provider into an intelligent classroom co-teacher that helps educators create engaging, interactive, and effective learning experiences.

---

# 🚨 Problem Statement

Teachers often face multiple challenges during live classroom sessions:

* Explaining complex concepts to students with different learning abilities.
* Repeating explanations multiple times.
* Creating quizzes during class.
* Keeping students engaged.
* Ensuring students truly understand concepts.
* Managing classroom flow while teaching.
* Simplifying difficult terminology.
* Using smart boards effectively during lessons.

Existing AI tools are typically designed as question-answer systems and do not support actual classroom teaching workflows.

---

# ❌ Limitations of Existing Solutions

Most educational AI tools follow this workflow:

Teacher → AI Answer → End

Problems:

* No understanding verification
* No active student engagement
* No classroom-focused interface
* No teacher-centered workflow
* Generic explanations
* Difficult terms remain confusing
* No real-time teaching support

These systems provide information but do not actively support learning.

---

# ✅ Proposed Solution

MentorAI acts as an AI-powered classroom co-pilot that assists teachers during live classroom sessions.

Teachers can:

* Explain concepts instantly
* Generate quizzes through voice commands
* Simplify difficult terms
* Present visual learning content
* Create story-based explanations
* Check student understanding through AI-generated questions

The entire experience is optimized for classroom teaching rather than individual chatbot conversations.

---

# 🚀 Core Features

## 📖 Live Concept Simplification

Teachers can explain any concept using voice or text.

Example:

> Explain Photosynthesis

MentorAI automatically generates:

* Simplified explanation
* Key learning points
* Difficult term explanations
* Visual summaries
* Follow-up understanding questions

### Benefits

* Saves teaching time
* Improves clarity
* Supports different learning levels

---

## 🧠 Socratic Teaching Engine

### The Problem

Students often listen passively without demonstrating understanding.

### The Solution

After every explanation, MentorAI automatically generates a conceptual follow-up question.

Example:

Teacher:

> Explain Photosynthesis

MentorAI:

> Plants use sunlight to make food.

MentorAI:

> What do you think would happen if sunlight was unavailable?

### Impact

* Encourages critical thinking
* Promotes classroom interaction
* Verifies understanding

This is the primary innovation of MentorAI.

---

## 📚 Textbook-Aware Learning

Teachers can upload:

* PDF Textbooks
* Notes
* Worksheets

MentorAI prioritizes uploaded educational materials while generating explanations.

Benefits:

* Content consistency
* Context-aware teaching
* Better lesson alignment

---

## 📝 Voice Triggered Quiz Generation

Teachers can instantly create quizzes.

Example:

> Create a quiz on Photosynthesis

MentorAI generates:

* Multiple Choice Questions
* Answer Options
* Correct Answers
* Difficulty Levels

### Benefits

* Saves classroom time
* Enables instant assessment
* Improves student participation

---

## 🔍 Confusion Detector

### The Problem

Students often struggle with technical terms.

### The Solution

MentorAI automatically identifies difficult words and explains them in simple language.

Example:

| Difficult Term | Simplified Meaning              |
| -------------- | ------------------------------- |
| Photosynthesis | Process plants use to make food |
| Chlorophyll    | Green substance in leaves       |
| Carbon Dioxide | Gas found in air                |

### Benefits

* Reduces confusion
* Improves accessibility
* Supports diverse learners

---

## 📖 Story Mode Learning

Teachers can request:

> Explain as a Story

MentorAI converts concepts into engaging educational narratives.

Example:

"One morning, a plant woke up hungry and looked toward the sun..."

### Benefits

* Better retention
* Increased engagement
* Easier understanding of abstract concepts

---

## 🎤 Voice-Enabled Workflow

MentorAI supports natural classroom interactions.

Supported Languages:

* English
* Hindi
* Hinglish

Teachers can operate the system without interrupting their lesson flow.

---

# 🖥️ Dual-Pane Classroom Experience

Traditional AI systems are designed for individuals.

MentorAI is designed for classrooms.

---

## 👨‍🏫 Teacher Command Center

Displays:

* Voice Transcript
* Uploaded Documents
* Current Topic
* Chapter Information
* AI Controls
* Quiz Controls
* Story Mode Controls

Visible only to the teacher.

---

## 👩‍🎓 Student Learning Board

Displays:

* Explanation
* Key Points
* Difficult Terms
* Visual Summaries
* Story Mode Output
* Quiz Questions
* Socratic Questions

Optimized for classroom smart boards.

---

# 🏗️ System Architecture

```text
Teacher Input
      │
      ▼
Speech-to-Text (Whisper)
      │
      ▼
Intent Detection
      │
      ▼
Document Retrieval Layer
      │
      ▼
MentorAI Teaching Engine
      │
 ┌────┼────┐
 ▼    ▼    ▼
Explain Quiz Story
      │
      ▼
Student Learning Board
```

---

# 🛠️ Technology Stack

## Frontend

* Streamlit

Purpose:

* Teacher Dashboard
* Student Learning Board
* Interactive UI

---

## Backend

* Python

Purpose:

* Application Logic
* Prompt Handling
* Data Processing

---

## Large Language Model

### Gemini 2.5 Flash

Used For:

* Concept Simplification
* Quiz Generation
* Story Mode
* Socratic Questions
* Confusion Detection

Alternative Providers:

* OpenRouter
* Ollama

---

## Speech Recognition

### OpenAI Whisper

Used For:

* Voice Commands
* Hinglish Support
* Speech-to-Text Conversion

---

## Text-to-Speech

### gTTS

Used For:

* Audio Explanations
* Voice Feedback

---

## Document Processing

### PyPDF2

Used For:

* Textbook Uploads
* Notes Processing
* Content Retrieval

---

## Visualization

### Mermaid Diagrams

Used For:

* Concept Flow Diagrams
* Visual Learning Aids

---

## Deployment

### Streamlit Community Cloud

Alternative:

* Hugging Face Spaces

---

# 📂 Project Structure

```text
MentorAI/
│
├── app.py
├── prompts/
│   ├── explain_prompt.py
│   ├── quiz_prompt.py
│   └── story_prompt.py
│
├── services/
│   ├── ai_service.py
│   ├── speech_service.py
│   └── pdf_service.py
│
├── uploads/
│
├── assets/
│
├── requirements.txt
│
└── README.md
```

---

# 🎯 Expected Outcomes

## For Teachers

* Reduced lesson preparation effort
* Faster content generation
* Improved classroom engagement
* Hands-free teaching support

---

## For Students

* Better concept understanding
* Interactive learning
* Improved retention
* Reduced confusion

---

## For Schools

* Smarter classroom experiences
* Better utilization of smart boards
* AI-assisted teaching support

---

# 🔒 What We Are Not Building

To maintain focus and reliability, MentorAI intentionally does not include:

* Attendance Tracking
* Student Grading
* Learning Management Systems
* Parent Dashboards
* Social Features
* External Web Search
* Student Performance Analytics

The platform focuses exclusively on improving classroom teaching and learning.

---

# 🌟 Key Innovations

✅ Socratic Teaching Engine

✅ Confusion Detector

✅ Story-Based Learning

✅ Voice-Enabled Teaching

✅ Textbook-Aware Explanations

✅ Classroom-Centric Dual Pane Interface

✅ Interactive Learning Experience

---

# 📈 Educational Impact

MentorAI transforms classroom learning from:

```text
Teacher
   ↓
Explanation
   ↓
End
```

to:

```text
Teacher
   ↓
AI Explanation
   ↓
Student Interaction
   ↓
Understanding Check
   ↓
Active Learning
```

The result is a more engaging, interactive, and effective classroom experience.

---

# 💡 Guiding Principle

> AI should not only provide answers.
>
> AI should help teachers create understanding.

---

# 👨‍💻 Developed By

**Poojha M**

B.Tech Artificial Intelligence & Data Science

---

### Built with ❤️ for Smarter Classrooms
