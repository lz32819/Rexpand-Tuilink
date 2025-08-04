import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import ClassifierResult, TopicSuggesterResult
from utils.llm import invoke_llm


def suggest_topics(
    context: Context, classified_category: ClassifierResult, dry_run: bool = False
) -> TopicSuggesterResult:
    system_prompt = f"""\
You are a helpful assistant that suggests topics for the job seeker to reply for a potential referral or intro call.
You will be given the conversation history, classified category, job seeker profile (optional), and referrer profile (optional).
You will need to provide topics along with confidence score and reason.
When suggesting topics, always evaluate the latest messages first.
Never make up facts.

Topic Examples (but not limited to):
1. Thank you
2. Express interest
3. Express background match
4. Ask for a brief call
5. Ask for alternative referrers
6. Follow-up
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}
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
        "name": "topic_suggester_result",
        "strict": True,
        "type": "json_schema",
        "schema": TopicSuggesterResult.model_json_schema(),
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

    return TopicSuggesterResult(**json.loads(response.content[0]["text"]))
