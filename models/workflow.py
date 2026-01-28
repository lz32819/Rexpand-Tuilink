from models.base import BaseModel
from models.context import Context
from models.llm_result import (
    ClassifierResult,
    MessageGeneratorResult,
    TopicSuggesterResult,
    ActionsSummaryResult,
    ReferralPossibilityResult,
    CompletedAction,
)


class State(BaseModel):
    step: str | None = None
    context: Context
    classified_category: ClassifierResult | None = None
    # Cached routing decisions (computed from classified_category + category metadata)
    reply_needed: bool | None = None
    human_action_required: bool | None = None
    actions_fulfilled: bool | None = None
    # Used to automatically reset cached fields when the underlying conversation changes.
    context_fingerprint: str | None = None
    suggested_topics: TopicSuggesterResult | None = None
    selected_topics: TopicSuggesterResult | None = None
    generated_reply_message: MessageGeneratorResult | None = None
    actions_summary: ActionsSummaryResult | None = None
    referral_possibility: ReferralPossibilityResult | None = None
    questions_exist: bool | None = None
    questions_answered: bool | None = None
    completed_actions: list[CompletedAction] = []
