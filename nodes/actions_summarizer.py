import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import ClassifierResult, ActionsSummaryResult
from utils.llm import invoke_llm


def summarize_actions(
    context: Context,
    classified_category: ClassifierResult,
    dry_run: bool = False,
) -> ActionsSummaryResult:
    system_prompt = f"""\
You are a helpful assistant that analyzes conversations and summarizes the required human actions.
You will be given the existing conversation messages and classified conversation category.
You need to identify what specific actions the human needs to take and provide a clear summary.
For each action, provide:
- A clear, actionable description
- Priority level (high, medium, low)
- Description of why this action is needed
- Referenced message IDs that support this action requirement

Never make up facts. Base your analysis only on the provided conversation context.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}

Please analyze this conversation and identify what human actions are required.
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
        "name": "actions_summary_result",
        "strict": True,
        "type": "json_schema",
        "schema": ActionsSummaryResult.model_json_schema(),
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

    return ActionsSummaryResult(**json.loads(response.content[0]["text"]))
