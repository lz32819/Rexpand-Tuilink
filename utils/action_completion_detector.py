from models.llm_result import ActionsSummaryResult


def are_all_actions_completed(
    actions_summary: ActionsSummaryResult,
) -> bool:
    """
    Check if all actions from the actions_summary have been completed.
    """
    if not actions_summary or not actions_summary.actions:
        return True

    return all(bool(getattr(a, "completed", False)) for a in actions_summary.actions)
