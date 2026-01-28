from models.base import BaseModel


class Category(BaseModel):
    # Group label for organizing categories (e.g. "explicit_agreement", "no_response").
    category_group: str
    category: str
    description: str
    clarification: str


class ExtendedCategory(Category):
    human_action_required: bool
    reply_needed: bool
