from models.base import BaseModel


class Category(BaseModel):
    category: str
    description: str
    clarification: str


class ExtendedCategory(Category):
    human_action_required: bool
    reply_needed: bool
