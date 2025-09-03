import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import ActionsSummaryResult, CompletedAction
from utils.llm import invoke_llm


def detect_completed_actions(
    context: Context, 
    actions_summary: ActionsSummaryResult, 
    existing_completed_actions: list[CompletedAction] = None,
    dry_run: bool = False
) -> list[CompletedAction]:
    """
    Detect which actions from the actions_summary have been completed by the user.
    Returns a list of CompletedAction objects for newly completed actions.
    """
    if existing_completed_actions is None:
        existing_completed_actions = []
    
    # Get action IDs that are already completed
    completed_action_ids = {action.action_id for action in existing_completed_actions}
    
    system_prompt = """\
You are an action completion analyzer that determines which actions have been completed by the user.

Your task is to:
1. Analyze the conversation history to identify which actions have been completed
2. Only return actions that are NEWLY completed (not already in the existing_completed_actions list)
3. For each completed action, provide:
   - A unique action_id (based on the action description)
   - The action description
   - Whether it's completed (true/false)

Rules:
- Only consider actions that are explicitly mentioned in the actions_summary
- An action is considered completed if the user has taken concrete steps to fulfill it
- Look for evidence in the conversation that shows the action was completed
- Be conservative - only mark actions as completed if there's clear evidence
- Generate action_id as a hash or unique identifier based on the action description

Return your analysis as JSON with the following structure:
{
    "completed_actions": [
        {
            "action_id": "unique_identifier",
            "action_description": "description of the action",
            "completed": true
        }
    ]
}
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Actions Summary:
{actions_summary}

Existing Completed Actions (already completed):
{existing_completed_actions}

Please analyze which actions from the actions_summary have been newly completed by the user.
Only return actions that are not already in the existing_completed_actions list.
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
        "name": "action_completion_result",
        "strict": True,
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "completed_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_id": {"type": "string"},
                            "action_description": {"type": "string"},
                            "completed": {"type": "boolean"}
                        },
                        "required": ["action_id", "action_description", "completed"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["completed_actions"],
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

    result = json.loads(response.content[0]["text"])
    
    # Convert to CompletedAction objects
    completed_actions = []
    for action_data in result["completed_actions"]:
        completed_action = CompletedAction(
            action_id=action_data["action_id"],
            action_description=action_data["action_description"],
            completed=action_data["completed"]
        )
        completed_actions.append(completed_action)
    
    return completed_actions


def are_all_actions_completed(
    actions_summary: ActionsSummaryResult, 
    completed_actions: list[CompletedAction]
) -> bool:
    """
    Check if all actions from the actions_summary have been completed.
    """
    if not actions_summary or not actions_summary.actions:
        return True
    
    # Count how many actions are marked as completed
    completed_count = sum(1 for action in completed_actions if action.completed)
    
    # For now, we'll assume all actions are completed if we have any completed actions
    # This is a simplified approach - in a real system, you'd want to match
    # the actions more precisely
    return completed_count >= len(actions_summary.actions)
