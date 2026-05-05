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
1.A clear, actionable description
2.Priority level (high, medium, low)
3.Description of why this action is needed
4.Referenced message IDs that support this action requirement
5.Input type:
   - Use "message_response" when the user needs to reply with text, especially when the referrer asked a question.
   - Use "completion_boolean" when the user needs to do something outside the chat and later confirm completion, such as sending a resume or sharing a link.

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
        # NOTE: We intentionally omit runtime-managed fields like `Action.completed`,
        # `Action.response_text`, and `Action.response_recorded` here. OpenAI "strict"
        # JSON schema requires `required` to include every key in `properties`, while
        # these fields are populated later by the client/handler.
        "schema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "priority": {"type": "string"},
                            "description": {"type": "string"},
                            "input_type": {
                                "type": "string",
                                "enum": ["message_response", "completion_boolean"],
                            },
                            "referenced_message_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "action",
                            "priority",
                            "description",
                            "input_type",
                            "referenced_message_ids",
                        ],
                        "additionalProperties": False,
                    },
                },
                "summary": {"type": "string"},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["actions", "summary", "confidence", "reason"],
            "additionalProperties": False,
        },
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
