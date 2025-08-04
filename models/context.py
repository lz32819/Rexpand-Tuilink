from typing import Literal

from models.base import BaseModel


class UserProfile(BaseModel):
    name: str


class ReferrerProfile(BaseModel):
    name: str


class ConversationMessage(BaseModel):
    id: str
    role: Literal["job seeker", "referrer"]
    name: str
    body_text: str
    attachments_text: str
    delivered_at: int


class Context(BaseModel):
    messages: list[ConversationMessage]
    user_profile: UserProfile | None = None
    referrer_profile: ReferrerProfile | None = None
