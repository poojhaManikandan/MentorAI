# llm.py — Classroom Co-Pilot AI (Enhanced Edition)
# ─────────────────────────────────────────────────────────────────────────────
# Gemini 2.5 Flash LLM integration.
#
# Key Features:
#   • Textbook-Grounded (RAG via provided context string)
#   • Class-Adaptive (Class 1 to 12 complexity scaling)
#   • Hinglish output (Roman script mix of Hindi + English)
#   • Structured Pydantic JSON schemas
#   • Socratic Engine & Confusion Detector
#   • 3-Level Understanding Check (Easy, Medium, Hard)
#   • Quiz Engine (5 MCQs with difficulty levels)
# ─────────────────────────────────────────────────────────────────────────────

import json
import requests
from typing import List, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC OUTPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class ConfusingTerm(BaseModel):
    term: str = Field(description="The difficult English/scientific term")
    simple_meaning: str = Field(description="A very simple Hinglish meaning using a relatable Indian analogy (1 sentence)")


class UnderstandingQuestion(BaseModel):
    level: str = Field(description="Must be 'Easy', 'Medium', or 'Hard'")
    question: str = Field(description="A brief question in Hinglish to check student understanding.")
    expected_answer_hint: str = Field(description="Brief hint for the teacher in English.")


class QuizItem(BaseModel):
    difficulty: str = Field(description="Must be 'Easy', 'Medium', or 'Hard'")
    question: str = Field(description="The question in English or Hinglish")
    options: List[str] = Field(description="Exactly 4 answer options")
    answer: str = Field(description="The exact text of the correct option from the list above")
    explanation: Optional[str] = Field(None, description="Brief Hinglish explanation of why this answer is correct")


class ClassroomContent(BaseModel):
    """
    Master response schema for all classroom content types.
    """
    intent: str = Field(description="CONCEPT | STORY | QUIZ")
    topic: str = Field(description="The core topic")
    class_level: str = Field(description="The target audience (e.g., 'Class 6')")

    # ── CONCEPT / STORY fields ──────────────────────────────────────────────
    explanation: Optional[str] = Field(
        None,
        description="2-3 paragraph Hinglish explanation. Bold key terms. Use textbook context if provided. Empty for QUIZ."
    )
    story: Optional[str] = Field(
        None,
        description="A rich narrative Hinglish story that teaches the concept. Only for STORY intent."
    )
    key_points: Optional[List[str]] = Field(
        None,
        description="3-5 key learning points in Hinglish. Empty for QUIZ."
    )
    confusing_terms: Optional[List[ConfusingTerm]] = Field(
        None,
        description="3-5 difficult terms with simple Hinglish meanings. Empty for QUIZ."
    )
    visual_summary: Optional[str] = Field(
        None,
        description="Valid Mermaid.js graph TD code representing the concept map. Double-quote all labels. Empty for QUIZ."
    )
    socratic_question: Optional[str] = Field(
        None,
        description="One Socratic follow-up question in Hinglish. Open-ended. Empty for QUIZ."
    )
    understanding_check: Optional[List[UnderstandingQuestion]] = Field(
        None,
        description="Exactly 3 questions (1 Easy, 1 Medium, 1 Hard) to check student understanding. Empty for QUIZ."
    )

    # ── QUIZ fields ─────────────────────────────────────────────────────────
    quiz: Optional[List[QuizItem]] = Field(
        None,
        description="Exactly 5 MCQs (mix of Easy, Medium, Hard). Only for QUIZ intent."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT & REST SCHEMA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """
You are "MentorAI" — a deeply intelligent AI co-teacher for Indian classrooms.
Your primary job is to teach, not just answer. Every output must be structured to improve student understanding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 LANGUAGE & TONE: HINGLISH (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL explanations, stories, key points, and questions MUST be in Hinglish (a natural mix of Hindi and English written in Roman script, NEVER Devanagari).
Tone: Warm, enthusiastic, and educational.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 TEXTBOOK GROUNDING (RAG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If TEXTBOOK CONTENT is provided in the prompt, you MUST use it as your primary source of truth.
- Do NOT contradict the textbook.
- Extract facts, definitions, and examples from the provided textbook text.
- If the textbook does not contain the answer, rely on your general educational knowledge but keep it simple.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎓 CLASS ADAPTIVE LEARNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adjust the complexity of vocabulary and explanation to the target Class Level.
For lower classes (e.g. Class 1-5), use extremely simple Hinglish and very simple everyday analogies.
For middle classes (e.g. Class 6-8), use slightly more advanced Hinglish with relatable analogies.
For higher classes (e.g. Class 9-12), use standard terms with technical explanations but still simplified.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 INTENT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- CONCEPT: Populate 'explanation', 'key_points', 'confusing_terms', 'visual_summary', 'socratic_question', 'understanding_check'. Keep 'story' and 'quiz' as null.
- STORY: Populate 'story' (narrative Hinglish story), 'key_points', 'confusing_terms', 'visual_summary', 'socratic_question', 'understanding_check'. Keep 'explanation' and 'quiz' as null.
- QUIZ: Populate 'quiz' (exactly 5 items). All other fields should be null.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ SPECIAL COMMAND ADAPTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Format the main 'explanation' field based on the detected command focus to make it completely distinct:
1. If the command asks for a "Socratic Question" or "to make students think":
   - The 'explanation' field MUST act as an inquiry guide. Instead of standard answers, it should introduce the topic as a big puzzle or mystery, prompting students to think about 'why' or 'how' in Hinglish.
2. If the command asks for a "Debate" or "Debate Mode":
   - The 'explanation' field MUST present a controversial debate topic, highlighting two clear, opposing points of view (e.g. "Perspective A (Fayde) vs Perspective B (Nuksan)") in Hinglish.
3. If the command asks for a "Real-world Example":
   - The 'explanation' field MUST focus entirely on describing a highly relatable, real-world scenario (preferably with an Indian context like pressure cookers, train stations, local food, or cricket) to demonstrate the concept in action.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ CONTENT QUALITY STANDARDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CONFUSING TERMS: Provide 3-5 terms. The simple_meaning should be a short Hinglish explanation using relatable analogies.
2. VISUAL SUMMARY: Provide valid Mermaid.js graph TD code representing the concept map. Emojis can be included. Always double-quote all labels: A["label"].
3. SOCRATIC QUESTION: A single open-ended, thought-provoking Socratic question in Hinglish.
4. UNDERSTANDING CHECK: Exactly 3 questions (1 Easy, 1 Medium, 1 Hard) in Hinglish with EXPECTED ANSWER HINTS for the teacher.
5. QUIZ MCQs: Exactly 5 items. Option list must have exactly 4 strings. The answer field must match the correct option EXACTLY.
"""

# Rest API Schema for Gemini Structured Output
_REST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {"type": "STRING", "description": "CONCEPT | STORY | QUIZ"},
        "topic": {"type": "STRING", "description": "The topic name"},
        "class_level": {"type": "STRING", "description": "e.g., Class 6"},
        "explanation": {"type": "STRING", "description": "Simplified Hinglish explanation (empty for QUIZ)"},
        "story": {"type": "STRING", "description": "Story format explanation (only for STORY)"},
        "key_points": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "3-5 key points in Hinglish"
        },
        "confusing_terms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "term": {"type": "STRING"},
                    "simple_meaning": {"type": "STRING"}
                },
                "required": ["term", "simple_meaning"]
            }
        },
        "visual_summary": {"type": "STRING", "description": "Mermaid.js graph TD code"},
        "socratic_question": {"type": "STRING", "description": "One Socratic question in Hinglish"},
        "understanding_check": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "level": {"type": "STRING", "description": "Easy | Medium | Hard"},
                    "question": {"type": "STRING"},
                    "expected_answer_hint": {"type": "STRING"}
                },
                "required": ["level", "question", "expected_answer_hint"]
            }
        },
        "quiz": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "difficulty": {"type": "STRING", "description": "Easy | Medium | Hard"},
                    "question": {"type": "STRING"},
                    "options": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "answer": {"type": "STRING"},
                    "explanation": {"type": "STRING"}
                },
                "required": ["difficulty", "question", "options", "answer"]
            }
        }
    },
    "required": ["intent", "topic", "class_level"]
}


# ═══════════════════════════════════════════════════════════════════════════════
# SOCRATIC CONVERSATION SCHEMAS & PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

_SOCRATIC_SYSTEM_PROMPT = """
You are "MentorAI" — a Socratic co-teacher for Indian classrooms.
Your role is to guide students to understand concepts by asking questions, not by giving direct answers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 LANGUAGE & TONE: HINGLISH (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All feedback and questions MUST be in natural Hinglish (Roman script mix of Hindi and English, NEVER Devanagari).
Example: "Bilkul sahi socha aapne! Lekin agar hum..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 SOCRATIC METHOD RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. DO NOT give direct answers or definitions.
2. If the student's answer is correct or close: Praise them briefly, then ask a follow-up question that deepens their understanding.
3. If the student is incorrect or confused: Do not say they are wrong. Instead, present a simple, relatable daily-life analogy (preferably Indian context, like pressure cookers, traffic, tea/chai, local markets) that highlights the gap in their logic, then ask a simpler guiding question.
4. Keep guidance warm, encouraging, and concise (2-3 sentences max).
5. Always generate a 'next_question' that is open-ended.
"""

_SOCRATIC_REST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "guidance": {
            "type": "STRING",
            "description": "Encouraging Hinglish feedback or analogy guiding the student. Do NOT give the direct answer."
        },
        "next_question": {
            "type": "STRING",
            "description": "Next open-ended Hinglish Socratic question."
        }
    },
    "required": ["guidance", "next_question"]
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_classroom_content(
    provider: str,
    model_name: str,
    api_key: str = "",
    command: str = "",
    class_level: str = "",
    context: str = "",
    subject: str = "",
    chapter: str = "",
    topic: str = "",
) -> ClassroomContent:
    """
    Generates structured Hinglish classroom content using Gemini API, OpenRouter, or Ollama.
    """
    if provider == "Gemini":
        return _generate_gemini(
            api_key=api_key,
            model_name=model_name,
            command=command,
            class_level=class_level,
            context=context,
            subject=subject,
            chapter=chapter,
            topic=topic
        )
    elif provider == "OpenRouter":
        return _generate_openrouter(
            api_key=api_key,
            model_name=model_name,
            command=command,
            class_level=class_level,
            context=context,
            subject=subject,
            chapter=chapter,
            topic=topic
        )
    else:
        return _generate_ollama(
            model_name=model_name,
            command=command,
            class_level=class_level,
            context=context,
            subject=subject,
            chapter=chapter,
            topic=topic
        )


def _sanitize_classroom_content_data(parsed_data: dict, command: str, topic: str, class_level: str) -> dict:
    """Safeguards mandatory Pydantic fields and cleans up malformed nested structures."""
    
    # 1. Mandatory base fields
    if "intent" not in parsed_data or not parsed_data["intent"]:
        if "quiz" in command.lower():
            parsed_data["intent"] = "QUIZ"
        elif "story" in command.lower():
            parsed_data["intent"] = "STORY"
        else:
            parsed_data["intent"] = "CONCEPT"
            
    if "topic" not in parsed_data or not parsed_data["topic"]:
        parsed_data["topic"] = topic if topic else "General Topic"
        
    if "class_level" not in parsed_data or not parsed_data["class_level"]:
        parsed_data["class_level"] = class_level if class_level else "Class 6"

    # Ensure other mandatory fields are strings
    parsed_data["intent"] = str(parsed_data["intent"]).upper()
    parsed_data["topic"] = str(parsed_data["topic"])
    parsed_data["class_level"] = str(parsed_data["class_level"])

    # 2. Key points list sanitization
    if "key_points" in parsed_data:
        kp = parsed_data["key_points"]
        if isinstance(kp, list):
            # Keep only non-empty strings
            cleaned_kp = [str(x).strip() for x in kp if x is not None and str(x).strip()]
            parsed_data["key_points"] = cleaned_kp if cleaned_kp else None
        else:
            parsed_data["key_points"] = None

    # 3. Confusing terms list sanitization
    if "confusing_terms" in parsed_data:
        ct = parsed_data["confusing_terms"]
        if isinstance(ct, list):
            cleaned_ct = []
            for item in ct:
                if isinstance(item, dict) and "term" in item and item["term"]:
                    term_str = str(item["term"]).strip()
                    meaning_str = str(item.get("simple_meaning", "Details pending")).strip()
                    cleaned_ct.append({"term": term_str, "simple_meaning": meaning_str})
            parsed_data["confusing_terms"] = cleaned_ct if cleaned_ct else None
        else:
            parsed_data["confusing_terms"] = None

    # 4. Understanding check list sanitization
    if "understanding_check" in parsed_data:
        uc = parsed_data["understanding_check"]
        if isinstance(uc, list):
            cleaned_uc = []
            for item in uc:
                if isinstance(item, dict) and "question" in item and item["question"]:
                    level = str(item.get("level", "Easy")).strip()
                    q = str(item["question"]).strip()
                    hint = str(item.get("expected_answer_hint", "Answer details")).strip()
                    cleaned_uc.append({"level": level, "question": q, "expected_answer_hint": hint})
            parsed_data["understanding_check"] = cleaned_uc if cleaned_uc else None
        else:
            parsed_data["understanding_check"] = None

    # 5. Quiz items list sanitization
    if "quiz" in parsed_data:
        qz = parsed_data["quiz"]
        if isinstance(qz, list):
            cleaned_qz = []
            for item in qz:
                if isinstance(item, dict) and "question" in item and item["question"]:
                    diff = str(item.get("difficulty", "Easy")).strip()
                    q = str(item["question"]).strip()
                    opts = item.get("options", [])
                    if not isinstance(opts, list) or len(opts) < 4:
                        opts = ["Option A", "Option B", "Option C", "Option D"]
                    else:
                        opts = [str(o).strip() for o in opts]
                    ans = str(item.get("answer", opts[0])).strip()
                    exp = str(item.get("explanation", "")).strip()
                    cleaned_qz.append({
                        "difficulty": diff,
                        "question": q,
                        "options": opts,
                        "answer": ans,
                        "explanation": exp
                    })
            parsed_data["quiz"] = cleaned_qz if cleaned_qz else None
        else:
            parsed_data["quiz"] = None

    return parsed_data


def _build_full_prompt(
    command: str,
    class_level: str,
    subject: str = "",
    chapter: str = "",
    topic: str = "",
    context: str = ""
) -> str:
    prompt_parts = [
        f"TARGET CLASS LEVEL: {class_level}",
        f"CONTEXT SUBJECT: {subject}" if subject else "",
        f"CONTEXT CHAPTER: {chapter}" if chapter else "",
        f"CURRENT TOPIC: {topic}" if topic else "",
        f"TEACHER COMMAND: {command}",
    ]
    if context.strip():
        prompt_parts.append(f"\n{context}")

    cmd_lower = command.lower()
    if "socratic" in cmd_lower or "think" in cmd_lower:
        prompt_parts.append("""
CRITICAL REQUIREMENT FOR SOCRATIC QUESTIONING MODE:
- The 'explanation' field MUST act as an inquiry guide. Instead of standard answers, it should introduce the topic as a big puzzle or mystery, prompting students to think about 'why' or 'how' in Hinglish. Do NOT explain the concept directly.
- The 'socratic_question' field MUST be populated with a thought-provoking open-ended question in Hinglish.
""")
    elif "debate" in cmd_lower:
        prompt_parts.append("""
CRITICAL REQUIREMENT FOR DEBATE MODE:
- The 'explanation' field MUST present a controversial debate topic, highlighting two clear, opposing points of view (e.g. "Perspective A (Fayde) vs Perspective B (Nuksan)") in Hinglish.
""")
    elif "real-world" in cmd_lower or "example" in cmd_lower:
        prompt_parts.append("""
CRITICAL REQUIREMENT FOR REAL-WORLD EXAMPLE:
- The 'explanation' field MUST focus entirely on describing a highly relatable, real-world scenario (preferably with an Indian context like pressure cookers, train stations, local food, or cricket) to demonstrate the concept in action.
""")

    return "\n".join(filter(None, prompt_parts))


def _generate_gemini(
    api_key: str,
    model_name: str,
    command: str,
    class_level: str,
    context: str = "",
    subject: str = "",
    chapter: str = "",
    topic: str = "",
) -> ClassroomContent:
    if not api_key or api_key.strip().lower() in ("", "mock"):
        return _get_mock_content(command.lower())

    # Build prompt payload
    full_prompt = _build_full_prompt(command, class_level, subject, chapter, topic, context)
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": _SYSTEM_PROMPT + "\n\nCRITICAL: You must return valid JSON matching the schema specified in responseSchema. Do not include markdown code blocks or any other text."},
                {"text": full_prompt}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _REST_SCHEMA,
            "temperature": 0.75
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no response candidates.")
            
        content_parts = candidates[0].get("content", {}).get("parts", [])
        if not content_parts:
            raise RuntimeError("Gemini returned empty parts.")
            
        generated_json_text = content_parts[0].get("text", "").strip()
        
        parsed_data = json.loads(generated_json_text)
        parsed_data = _sanitize_classroom_content_data(parsed_data, command, topic, class_level)
        return ClassroomContent(**parsed_data)
        
    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"Gemini API connection error: {req_err}") from req_err
    except json.JSONDecodeError as json_err:
        raise RuntimeError(f"Gemini API returned invalid JSON: {json_err}") from json_err
    except Exception as exc:
        raise RuntimeError(f"Gemini generation error: {exc}") from exc


def _generate_ollama(
    model_name: str,
    command: str,
    class_level: str,
    context: str = "",
    subject: str = "",
    chapter: str = "",
    topic: str = "",
) -> ClassroomContent:
    if not model_name or model_name.strip().lower() in ("", "mock"):
        return _get_mock_content(command.lower())

    # Build the prompt payload
    full_prompt = _build_full_prompt(command, class_level, subject, chapter, topic, context)

    # We add a hint at the end of the system prompt to enforce JSON structure
    ollama_system_prompt = _SYSTEM_PROMPT + """

CRITICAL: You must return ONLY valid JSON matching this exact structure:
{
  "intent": "CONCEPT" or "STORY" or "QUIZ",
  "topic": "Topic name here",
  "class_level": "Class level here",
  "explanation": "Hinglish explanation here (null for QUIZ)",
  "story": "Hinglish story here (only if intent is STORY)",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "confusing_terms": [
    {"term": "Term 1", "simple_meaning": "Meaning in Hinglish"}
  ],
  "visual_summary": "graph TD\\n  A --> B",
  "socratic_question": "One open-ended Hinglish question",
  "understanding_check": [
    {"level": "Easy", "question": "Question 1", "expected_answer_hint": "Hint 1"},
    {"level": "Medium", "question": "Question 2", "expected_answer_hint": "Hint 2"},
    {"level": "Hard", "question": "Question 3", "expected_answer_hint": "Hint 3"}
  ],
  "quiz": null
}

Do not return markdown code blocks, preambles, or postambles. Just the JSON object.
"""

    payload = {
        "model": model_name,
        "format": "json",
        "system": ollama_system_prompt,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.75,
            "num_predict": 4096
        }
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        
        # Ollama returns the generated text in the 'response' field
        generated_json_text = data.get("response", "").strip()
        
        parsed_data = json.loads(generated_json_text)
        parsed_data = _sanitize_classroom_content_data(parsed_data, command, topic, class_level)
        return ClassroomContent(**parsed_data)

    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"Ollama connection error (is it running?): {req_err}") from req_err
    except json.JSONDecodeError as json_err:
        raise RuntimeError(f"Ollama returned invalid JSON: {json_err}") from json_err
    except Exception as exc:
        raise RuntimeError(f"Ollama generation error: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO CONTENT (For offline testing)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_mock_content(cmd: str) -> ClassroomContent:
    """Provides fallback demo content if no API key is set."""

    if "quiz" in cmd and "photosynthesis" in cmd:
        return ClassroomContent(
            intent="QUIZ",
            topic="Photosynthesis",
            class_level="Class 6",
            quiz=[
                QuizItem(difficulty="Easy", question="Plants apna khana kahan banate hain?", options=["Roots", "Stem", "Leaves", "Flowers"], answer="Leaves", explanation="Leaves plant ka kitchen hoti hain."),
                QuizItem(difficulty="Easy", question="Photosynthesis ke liye kaunsi gas zaroori hai?", options=["Oxygen", "Carbon Dioxide", "Nitrogen", "Hydrogen"], answer="Carbon Dioxide", explanation="Plants CO2 absorb karte hain aur O2 release karte hain."),
                QuizItem(difficulty="Medium", question="Leaves ka green color kis wajah se hota hai?", options=["Melanin", "Hemoglobin", "Chlorophyll", "Carotene"], answer="Chlorophyll", explanation="Chlorophyll sunlight ko capture karta hai."),
                QuizItem(difficulty="Medium", question="Plants hawa se CO2 kaise absorb karte hain?", options=["Roots se", "Stomata se", "Flowers se", "Bark se"], answer="Stomata se", explanation="Stomata leaves par chhote holes hote hain."),
                QuizItem(difficulty="Hard", question="Photosynthesis mein sunlight energy kis form mein convert hoti hai?", options=["Mechanical energy", "Heat energy", "Chemical energy", "Electrical energy"], answer="Chemical energy", explanation="Sunlight ki energy use karke glucose (chemical energy) banaya jata hai."),
            ]
        )

    if "photosynthesis" in cmd:
        return ClassroomContent(
            intent="CONCEPT",
            topic="Photosynthesis",
            class_level="Class 6",
            explanation="Photosynthesis ek aisi process hai jisse **green plants** sunlight, paani aur carbon dioxide ko use karke apna **khana khud banate hain**! Socho jaise plants ka apna solar-powered kitchen hai.\n\nIs process mein leaves mein mojud **Chlorophyll** sunlight ko absorb karta hai. Roots se paani aata hai, aur leaves ke chhote holes (**Stomata**) se CO2 aati hai. In sab se milkar **Glucose** banta hai, aur **Oxygen** hawa mein release hoti hai.",
            key_points=[
                "**Chlorophyll** sunlight ko absorb karta hai.",
                "**Stomata** se CO2 andar aati hai aur O2 bahar jati hai.",
                "Plants apna khana **Glucose** ke form mein banate hain."
            ],
            confusing_terms=[
                ConfusingTerm(term="Chlorophyll", simple_meaning="Leaves ka green color — yeh plant ka solar panel hai."),
                ConfusingTerm(term="Stomata", simple_meaning="Leaves ke neeche chhote holes — jaise plant ke nostrils."),
                ConfusingTerm(term="Glucose", simple_meaning="Ek simple sugar — plant ka ghar ka bana hua khana.")
            ],
            visual_summary='graph TD\n  A["☀️ Sunlight"] --> B["🌿 Chlorophyll"]\n  C["💧 Water"] --> B\n  D["💨 CO2"] --> B\n  B --> E["🍬 Glucose"]\n  B --> F["🌬️ Oxygen"]',
            socratic_question="Agar earth par ek mahine tak dhoop na nikle, toh sabse pehle kin living things par asar padega aur kyun?",
            understanding_check=[
                UnderstandingQuestion(level="Easy", question="Photosynthesis mein plant kaunsi gas release karte hain?", expected_answer_hint="Oxygen"),
                UnderstandingQuestion(level="Medium", question="Plant ka khana kis form mein banta hai?", expected_answer_hint="Glucose"),
                UnderstandingQuestion(level="Hard", question="Agar kisi plant ke leaves par dust jam jaye, toh photosynthesis par kya asar hoga?", expected_answer_hint="Stomata block ho jayenge, CO2 andar nahi aa payegi."),
            ]
        )

    raise ValueError("Demo Mode: Please enter a Gemini API Key to use custom topics, or try 'Explain Photosynthesis' or 'Create quiz on Photosynthesis'.")


# ═══════════════════════════════════════════════════════════════════════════════
# SOCRATIC DIALOGUE API CALLS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_socratic_response(
    provider: str,
    model_name: str,
    api_key: str = "",
    topic: str = "",
    class_level: str = "",
    context: str = "",
    initial_question: str = "",
    chat_history: List[dict] = None,
    student_reply: str = "",
) -> dict:
    if provider == "Gemini":
        return _generate_socratic_gemini(
            api_key=api_key,
            model_name=model_name,
            topic=topic,
            class_level=class_level,
            context=context,
            initial_question=initial_question,
            chat_history=chat_history,
            student_reply=student_reply
        )
    elif provider == "OpenRouter":
        return _generate_socratic_openrouter(
            api_key=api_key,
            model_name=model_name,
            topic=topic,
            class_level=class_level,
            context=context,
            initial_question=initial_question,
            chat_history=chat_history,
            student_reply=student_reply
        )
    else:
        return _generate_socratic_ollama(
            model_name=model_name,
            topic=topic,
            class_level=class_level,
            context=context,
            initial_question=initial_question,
            chat_history=chat_history,
            student_reply=student_reply
        )


def _generate_socratic_gemini(
    api_key: str,
    model_name: str,
    topic: str,
    class_level: str,
    context: str = "",
    initial_question: str = "",
    chat_history: List[dict] = None,
    student_reply: str = "",
) -> dict:
    if not api_key or api_key.strip().lower() in ("", "mock"):
        return _get_mock_socratic_response(topic, student_reply)

    # Build conversation context
    history_str = ""
    if chat_history:
        for msg in chat_history:
            role_name = "Student" if msg["role"] == "user" else "MentorAI"
            history_str += f"{role_name}: {msg['text']}\n"

    prompt_parts = [
        f"TOPIC: {topic}",
        f"TARGET CLASS LEVEL: {class_level}",
        f"TEXTBOOK CONTEXT: {context}" if context else "",
        f"INITIAL QUESTION: {initial_question}",
        f"CONVERSATION HISTORY:\n{history_str}" if history_str else "",
        f"NEW STUDENT REPLY: {student_reply}",
        "\nProvide your Socratic guidance and the next question in JSON format conforming to responseSchema."
    ]
    full_prompt = "\n".join(filter(None, prompt_parts))

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    payload = {
        "contents": [{
            "parts": [
                {"text": _SOCRATIC_SYSTEM_PROMPT + "\n\nCRITICAL: You must return valid JSON matching the schema specified in responseSchema. Do not include markdown code blocks or any other text."},
                {"text": full_prompt}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _SOCRATIC_REST_SCHEMA,
            "temperature": 0.75
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no response candidates for Socratic dialogue.")

        content_parts = candidates[0].get("content", {}).get("parts", [])
        if not content_parts:
            raise RuntimeError("Gemini returned empty parts for Socratic dialogue.")

        generated_json_text = content_parts[0].get("text", "").strip()
        parsed_data = json.loads(generated_json_text)
        return parsed_data

    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"Gemini API connection error: {req_err}") from req_err
    except json.JSONDecodeError as json_err:
        raise RuntimeError(f"Gemini API returned invalid JSON: {json_err}") from json_err
    except Exception as exc:
        raise RuntimeError(f"Gemini generation error: {exc}") from exc


def _generate_socratic_ollama(
    model_name: str,
    topic: str,
    class_level: str,
    context: str = "",
    initial_question: str = "",
    chat_history: List[dict] = None,
    student_reply: str = "",
) -> dict:
    if not model_name or model_name.strip().lower() in ("", "mock"):
        return _get_mock_socratic_response(topic, student_reply)

    # Build conversation context
    history_str = ""
    if chat_history:
        for msg in chat_history:
            role_name = "Student" if msg["role"] == "user" else "MentorAI"
            history_str += f"{role_name}: {msg['text']}\n"

    prompt_parts = [
        f"TOPIC: {topic}",
        f"TARGET CLASS LEVEL: {class_level}",
        f"TEXTBOOK CONTEXT: {context}" if context else "",
        f"INITIAL QUESTION: {initial_question}",
        f"CONVERSATION HISTORY:\n{history_str}" if history_str else "",
        f"NEW STUDENT REPLY: {student_reply}",
        "\nProvide Socratic guidance and next question in a JSON format: {\"guidance\": \"...\", \"next_question\": \"...\"}"
    ]
    full_prompt = "\n".join(filter(None, prompt_parts))

    ollama_system_prompt = _SOCRATIC_SYSTEM_PROMPT + "\n\nCRITICAL: You must return ONLY valid JSON matching the schema: {\"guidance\": \"...\", \"next_question\": \"...\"}. Do not include markdown code blocks or any other text."

    payload = {
        "model": model_name,
        "format": "json",
        "system": ollama_system_prompt,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.75,
            "num_predict": 1024
        }
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()

        generated_json_text = data.get("response", "").strip()
        parsed_data = json.loads(generated_json_text)
        return parsed_data

    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"Ollama connection error (is it running?): {req_err}") from req_err
    except json.JSONDecodeError as json_err:
        raise RuntimeError(f"Ollama returned invalid JSON: {json_err}") from json_err
    except Exception as exc:
        raise RuntimeError(f"Ollama generation error: {exc}") from exc


def _get_mock_socratic_response(topic: str, reply: str) -> dict:
    """Mock fallback for Socratic conversation dialogue."""
    return {
        "guidance": f"Aapne kaha '{reply}', yeh ek badhiya koshish hai! Lekin socho, kya isse topic '{topic}' ke saare main concepts cover ho rahe hain? Hum isse daily life mein aur kahan dekh sakte hain?",
        "next_question": f"Socho agar hum '{topic}' ko kisi simple everyday phenomenon se relate karein, toh aapko kya lagta hai iska sabse bada asar kahan padta hai?"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEBATE MODE DIALOGUE CONTEXT, SCHEMAS & PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

_DEBATE_SYSTEM_PROMPT = """
You are "MentorAI" — a highly engaging debate opponent and moderator for Indian classrooms.
The students are debating the topic: {topic} at a {class_level} complexity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 LANGUAGE & TONE: HINGLISH (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All counter-arguments and follow-up challenges MUST be in natural Hinglish (Roman script).
Example: "Aapka point kaafi solid hai, par socho agar hum..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🆚 DEBATE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Listen carefully to the student's argument.
2. Play the opposing side! If the student's argument supports Perspective A, counter-argue using Perspective B's logic. If they support Perspective B, counter-argue using Perspective A. If they are neutral, push them to pick a stand.
3. Keep the tone respectful, friendly, but intellectually challenging.
4. Use simple textbook grounding or everyday Indian analogies (like street food, local transport, cricket, household habits) to support your points.
5. Keep your response brief (2-3 sentences max) followed by exactly ONE follow-up challenge/question.
"""

_DEBATE_REST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "counter_argument": {
            "type": "STRING",
            "description": "Counter-argument or response to the student's argument in Hinglish, playing the opposing debater or a critical moderator. Do NOT agree too easily, keep it competitive but polite."
        },
        "next_challenge": {
            "type": "STRING",
            "description": "A follow-up challenge or question in Hinglish prompting the student to defend their stance."
        }
    },
    "required": ["counter_argument", "next_challenge"]
}


def generate_debate_response(
    provider: str,
    model_name: str,
    api_key: str = "",
    topic: str = "",
    class_level: str = "",
    context: str = "",
    debate_intro: str = "",
    chat_history: List[dict] = None,
    student_reply: str = "",
) -> dict:
    if provider == "Gemini":
        return _generate_debate_gemini(
            api_key=api_key,
            model_name=model_name,
            topic=topic,
            class_level=class_level,
            context=context,
            debate_intro=debate_intro,
            chat_history=chat_history,
            student_reply=student_reply
        )
    elif provider == "OpenRouter":
        return _generate_debate_openrouter(
            api_key=api_key,
            model_name=model_name,
            topic=topic,
            class_level=class_level,
            context=context,
            debate_intro=debate_intro,
            chat_history=chat_history,
            student_reply=student_reply
        )
    else:
        return _generate_debate_ollama(
            model_name=model_name,
            topic=topic,
            class_level=class_level,
            context=context,
            debate_intro=debate_intro,
            chat_history=chat_history,
            student_reply=student_reply
        )


def _generate_debate_gemini(
    api_key: str,
    model_name: str,
    topic: str,
    class_level: str,
    context: str = "",
    debate_intro: str = "",
    chat_history: List[dict] = None,
    student_reply: str = "",
) -> dict:
    if not api_key or api_key.strip().lower() in ("", "mock"):
        return _get_mock_debate_response(topic, student_reply)

    # Build conversation context
    history_str = ""
    if chat_history:
        for msg in chat_history:
            role_name = "Student" if msg["role"] == "user" else "MentorAI"
            history_str += f"{role_name}: {msg['text']}\n"

    prompt_parts = [
        f"DEBATE TOPIC INTRO: {debate_intro}",
        f"TARGET CLASS LEVEL: {class_level}",
        f"TEXTBOOK CONTEXT: {context}" if context else "",
        f"CONVERSATION HISTORY:\n{history_str}" if history_str else "",
        f"NEW STUDENT ARGUMENT: {student_reply}",
        "\nProvide your counter_argument and next_challenge in JSON format conforming to responseSchema."
    ]
    full_prompt = "\n".join(filter(None, prompt_parts))

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    payload = {
        "contents": [{
            "parts": [
                {"text": _DEBATE_SYSTEM_PROMPT.format(topic=topic, class_level=class_level) + "\n\nCRITICAL: You must return valid JSON matching the schema specified in responseSchema. Do not include markdown code blocks or any other text."},
                {"text": full_prompt}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _DEBATE_REST_SCHEMA,
            "temperature": 0.75
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no response candidates for debate.")

        content_parts = candidates[0].get("content", {}).get("parts", [])
        if not content_parts:
            raise RuntimeError("Gemini returned empty parts for debate.")

        generated_json_text = content_parts[0].get("text", "").strip()
        parsed_data = json.loads(generated_json_text)
        return parsed_data

    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"Gemini API connection error: {req_err}") from req_err
    except json.JSONDecodeError as json_err:
        raise RuntimeError(f"Gemini API returned invalid JSON: {json_err}") from json_err
    except Exception as exc:
        raise RuntimeError(f"Gemini generation error: {exc}") from exc


def _generate_debate_ollama(
    model_name: str,
    topic: str,
    class_level: str,
    context: str = "",
    debate_intro: str = "",
    chat_history: List[dict] = None,
    student_reply: str = "",
) -> dict:
    if not model_name or model_name.strip().lower() in ("", "mock"):
        return _get_mock_debate_response(topic, student_reply)

    # Build conversation context
    history_str = ""
    if chat_history:
        for msg in chat_history:
            role_name = "Student" if msg["role"] == "user" else "MentorAI"
            history_str += f"{role_name}: {msg['text']}\n"

    prompt_parts = [
        f"DEBATE TOPIC INTRO: {debate_intro}",
        f"TARGET CLASS LEVEL: {class_level}",
        f"TEXTBOOK CONTEXT: {context}" if context else "",
        f"CONVERSATION HISTORY:\n{history_str}" if history_str else "",
        f"NEW STUDENT ARGUMENT: {student_reply}",
        "\nProvide counter_argument and next_challenge in a JSON format: {\"counter_argument\": \"...\", \"next_challenge\": \"...\"}"
    ]
    full_prompt = "\n".join(filter(None, prompt_parts))

    ollama_system_prompt = _DEBATE_SYSTEM_PROMPT.format(topic=topic, class_level=class_level) + "\n\nCRITICAL: You must return ONLY valid JSON matching the schema: {\"counter_argument\": \"...\", \"next_challenge\": \"...\"}. Do not include markdown code blocks or any other text."

    payload = {
        "model": model_name,
        "format": "json",
        "system": ollama_system_prompt,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.75,
            "num_predict": 1024
        }
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()

        generated_json_text = data.get("response", "").strip()
        parsed_data = json.loads(generated_json_text)
        return parsed_data

    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"Ollama connection error (is it running?): {req_err}") from req_err
    except json.JSONDecodeError as json_err:
        raise RuntimeError(f"Ollama returned invalid JSON: {json_err}") from json_err
    except Exception as exc:
        raise RuntimeError(f"Ollama generation error: {exc}") from exc


def _get_mock_debate_response(topic: str, reply: str) -> dict:
    """Mock fallback for Debate conversation dialogue."""
    return {
        "counter_argument": f"Aapne kaha '{reply}', yeh ek interesting debate point hai! Lekin agar hum doosre side se dekhein, toh kya yeh hamare main concept '{topic}' ke basic guidelines ko satisfy karega?",
        "next_challenge": f"Aap is argument ke response mein doosre side ko kaise defend karenge? Koi solid point batayein."
    }


def _generate_openrouter(
    api_key: str,
    model_name: str,
    command: str,
    class_level: str,
    context: str = "",
    subject: str = "",
    chapter: str = "",
    topic: str = "",
) -> ClassroomContent:
    if not api_key or api_key.strip().lower() in ("", "mock"):
        return _get_mock_content(command.lower())

    api_key = api_key.strip()
    model_name = model_name.strip()

    full_prompt = _build_full_prompt(command, class_level, subject, chapter, topic, context)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mentor-ai",
        "X-Title": "MentorAI"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT + "\n\nCRITICAL: You must return valid JSON matching this exact structure, containing only the JSON. Do not write markdown tags or extra talk."},
            {"role": "user", "content": full_prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.75
    }

    generated_json_text = ""
    response = None
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no response choices.")

        generated_json_text = choices[0].get("message", {}).get("content", "").strip()
        
        if generated_json_text.startswith("```"):
            lines = generated_json_text.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            generated_json_text = "\n".join(lines).strip()
        if generated_json_text.lower().startswith("json"):
            generated_json_text = generated_json_text[4:].strip()

        parsed_data = json.loads(generated_json_text)
        parsed_data = _sanitize_classroom_content_data(parsed_data, command, topic, class_level)
        return ClassroomContent(**parsed_data)

    except requests.exceptions.RequestException as req_err:
        err_detail = str(req_err)
        if response is not None:
            try:
                err_detail += f" | Details: {response.text}"
            except Exception:
                pass
        raise RuntimeError(f"OpenRouter API connection error: {err_detail}") from req_err
    except json.JSONDecodeError as json_err:
        raise RuntimeError(f"OpenRouter returned invalid JSON: {json_err}\nRaw output: {generated_json_text}") from json_err
    except Exception as exc:
        raise RuntimeError(f"OpenRouter generation error: {exc}") from exc


def _generate_socratic_openrouter(
    api_key: str,
    model_name: str,
    topic: str,
    class_level: str,
    context: str = "",
    initial_question: str = "",
    chat_history: List[dict] = None,
    student_reply: str = "",
) -> dict:
    if not api_key or api_key.strip().lower() in ("", "mock"):
        return _get_mock_socratic_response(topic, student_reply)

    api_key = api_key.strip()
    model_name = model_name.strip()

    messages = [
        {"role": "system", "content": _SOCRATIC_SYSTEM_PROMPT + "\n\nCRITICAL: You must return valid JSON matching this exact structure: {\"guidance\": \"...\", \"next_question\": \"...\"}. Do not include markdown code blocks or any other text."}
    ]
    
    if context:
        messages.append({"role": "system", "content": f"TEXTBOOK CONTEXT:\n{context}"})
    
    messages.append({"role": "system", "content": f"TOPIC: {topic}\nTARGET CLASS LEVEL: {class_level}\nINITIAL SOCRATIC QUESTION: {initial_question}"})

    if chat_history:
        for msg in chat_history:
            messages.append({
                "role": "user" if msg["role"] == "user" else "assistant",
                "content": msg["text"]
            })

    messages.append({"role": "user", "content": student_reply})

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mentor-ai",
        "X-Title": "MentorAI"
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.75
    }

    generated_json_text = ""
    response = None
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no response choices for Socratic dialogue.")

        generated_json_text = choices[0].get("message", {}).get("content", "").strip()

        if generated_json_text.startswith("```"):
            lines = generated_json_text.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            generated_json_text = "\n".join(lines).strip()
        if generated_json_text.lower().startswith("json"):
            generated_json_text = generated_json_text[4:].strip()

        parsed_data = json.loads(generated_json_text)
        if "guidance" not in parsed_data:
            parsed_data["guidance"] = "Accha prayas!"
        if "next_question" not in parsed_data:
            parsed_data["next_question"] = "Aapko kya lagta hai?"
        return parsed_data

    except requests.exceptions.RequestException as req_err:
        err_detail = str(req_err)
        if response is not None:
            try:
                err_detail += f" | Details: {response.text}"
            except Exception:
                pass
        raise RuntimeError(f"OpenRouter Socratic API connection error: {err_detail}") from req_err
    except json.JSONDecodeError as json_err:
        raise RuntimeError(f"OpenRouter Socratic returned invalid JSON: {json_err}\nRaw output: {generated_json_text}") from json_err
    except Exception as exc:
        raise RuntimeError(f"OpenRouter Socratic generation error: {exc}") from exc


def _generate_debate_openrouter(
    api_key: str,
    model_name: str,
    topic: str,
    class_level: str,
    context: str = "",
    debate_intro: str = "",
    chat_history: List[dict] = None,
    student_reply: str = "",
) -> dict:
    if not api_key or api_key.strip().lower() in ("", "mock"):
        return _get_mock_debate_response(topic, student_reply)

    api_key = api_key.strip()
    model_name = model_name.strip()

    system_prompt = _DEBATE_SYSTEM_PROMPT.format(topic=topic, class_level=class_level)
    messages = [
        {"role": "system", "content": system_prompt + "\n\nCRITICAL: You must return ONLY valid JSON matching the schema: {\"counter_argument\": \"...\", \"next_challenge\": \"...\"}. Do not include markdown code blocks or any other text."}
    ]
    
    if context:
        messages.append({"role": "system", "content": f"TEXTBOOK CONTEXT:\n{context}"})
        
    messages.append({"role": "system", "content": f"DEBATE TOPIC INTRO: {debate_intro}"})

    if chat_history:
        for msg in chat_history:
            messages.append({
                "role": "user" if msg["role"] == "user" else "assistant",
                "content": msg["text"]
            })

    messages.append({"role": "user", "content": student_reply})

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mentor-ai",
        "X-Title": "MentorAI"
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.75
    }

    generated_json_text = ""
    response = None
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no response choices for debate.")

        generated_json_text = choices[0].get("message", {}).get("content", "").strip()

        if generated_json_text.startswith("```"):
            lines = generated_json_text.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            generated_json_text = "\n".join(lines).strip()
        if generated_json_text.lower().startswith("json"):
            generated_json_text = generated_json_text[4:].strip()

        parsed_data = json.loads(generated_json_text)
        if "counter_argument" not in parsed_data:
            parsed_data["counter_argument"] = "Aapka point sahi hai, par kya hum is doosri dasha ko ignore kar sakte hain?"
        if "next_challenge" not in parsed_data:
            parsed_data["next_challenge"] = "Aap iska kya uttar denge?"
        return parsed_data

    except requests.exceptions.RequestException as req_err:
        err_detail = str(req_err)
        if response is not None:
            try:
                err_detail += f" | Details: {response.text}"
            except Exception:
                pass
        raise RuntimeError(f"OpenRouter Debate API connection error: {err_detail}") from req_err
    except json.JSONDecodeError as json_err:
        raise RuntimeError(f"OpenRouter Debate returned invalid JSON: {json_err}\nRaw output: {generated_json_text}") from json_err
    except Exception as exc:
        raise RuntimeError(f"OpenRouter Debate generation error: {exc}") from exc
