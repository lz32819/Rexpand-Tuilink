from pydantic import ConfigDict

from models.base import BaseModel


class ClassifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    confidence: float
    reason: str
    referenced_message_ids: list[str]


class Topic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str
    confidence: float
    reason: str


class TopicSuggesterResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topics: list[Topic]


class MessageGeneratorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    confidence: float
    reason: str


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    priority: str
    description: str
    referenced_message_ids: list[str]


class ActionsSummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[Action]
    summary: str
    confidence: float
    reason: str


class ReferralPossibilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    referral_possible: bool
    confidence: float
    reason: str
    next_steps: list[str]
    barriers: list[str]
