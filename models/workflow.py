from models.base import BaseModel
from models.context import Context
from models.llm_result import (
    ClassifierResult,
    MessageGeneratorResult,
    TopicSuggesterResult,
)


class State(BaseModel):
    step: str | None = None
    context: Context
    classified_category: ClassifierResult | None = None
    suggested_topics: TopicSuggesterResult | None = None
    selected_topics: TopicSuggesterResult | None = None
    generated_reply_message: MessageGeneratorResult | None = None
