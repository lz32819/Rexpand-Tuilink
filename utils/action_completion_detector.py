from models.llm_result import ActionsSummaryResult


def are_all_actions_completed(
    actions_summary: ActionsSummaryResult,
) -> bool:
    """
    Check if all actions from the actions_summary have been completed.
    """
    if not actions_summary or not actions_summary.actions:
        return True

    def _is_completed(action) -> bool:
        if getattr(action, "input_type", None) == "message_response":
            return bool((getattr(action, "response_text", None) or "").strip())
        return bool(getattr(action, "completed", False))

    return all(_is_completed(action) for action in actions_summary.actions)
