import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from utils.llm import invoke_llm


def detect_questions(context: Context, dry_run: bool = False) -> dict:
    """
    Detect if the referrer has asked questions and if they have been answered.
    Returns a dict with 'questions_exist' and 'questions_answered' boolean values.
    """
    system_prompt = """\
You are a conversation analyzer that detects questions and answers in conversations between job seekers and referrers.

Your task is to:
1. Identify if the referrer has asked any questions to the job seeker
2. Determine if the job seeker has answered all the questions asked by the referrer

Rules:
- A question is considered answered if the job seeker has provided a substantive response
- Look at the conversation chronologically - questions asked later must be answered later
- If no questions exist, both questions_exist and questions_answered should be false
- If questions exist but not all are answered, questions_exist=true, questions_answered=false
- If questions exist and all are answered, questions_exist=true, questions_answered=true

Return your analysis as JSON with the following structure:
{
    "questions_exist": boolean,
    "questions_answered": boolean,
    "reason": "explanation of your analysis"
}
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Please analyze this conversation to determine:
1. Has the referrer asked any questions to the job seeker?
2. Has the job seeker answered all questions asked by the referrer?
"""

    if context.user_profile:
        user_prompt += f"""
Job Seeker Profile:
{context.user_profile}
"""

    if context.referrer_profile:
        user_prompt += f"""
Referrer Profile:
{context.referrer_profile}
"""

    output_schema = {
        "name": "question_detection_result",
        "strict": True,
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "questions_exist": {"type": "boolean"},
                "questions_answered": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["questions_exist", "questions_answered", "reason"],
            "additionalProperties": False
        }
    }

    if dry_run:
        return system_prompt, user_prompt, output_schema

    response = invoke_llm(
        input=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        text={"format": output_schema},
        use_cache=True,
    )

    return json.loads(response.content[0]["text"])
