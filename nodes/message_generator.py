import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import (
    ClassifierResult,
    MessageGeneratorResult,
    TopicSuggesterResult,
)
from utils.llm import invoke_llm


def generate_message(
    context: Context,
    classified_category: ClassifierResult,
    selected_topics: TopicSuggesterResult,
    dry_run: bool = False,
) -> MessageGeneratorResult:
    system_prompt = f"""\
You are a helpful assistant that generates a message for the job seeker to reply for a potential referral or intro call.
You will be given the existing conversation messages, classified conversation category, selected topics for new message, job seeker profile (optional), and referrer profile (optional).
You will need to generate a message along with confidence score and reason.
When generating the message, always evaluate the latest existing messages first.
Never make up facts.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}

Selected Topics:
{selected_topics}
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
        "name": "message_generator_result",
        "strict": True,
        "type": "json_schema",
        "schema": MessageGeneratorResult.model_json_schema(),
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

    return MessageGeneratorResult(**json.loads(response.content[0]["text"]))
