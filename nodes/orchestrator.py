from rexpand_pyutils_file import read_file

from models.category import Category, ExtendedCategory
from models.workflow import State
from nodes.classifier import classify_conversation
from nodes.message_generator import generate_message
from nodes.topic_suggester import suggest_topics


CATEGORIES = [Category(**category) for category in read_file("./input/categories.json")]
EXTENDED_CATEGORY_LOOKUP = {
    category["category"]: ExtendedCategory(**category)
    for category in read_file("./input/categories.json")
}


def orchestrate(
    state: State,
) -> State:
    # If the reply message is already generated, return the current state as final state
    if state.generated_reply_message is not None:
        state.step = "end: reply generated"
        return state

    # If the topics are already selected, generate the reply message and return the current state as final state
    if state.selected_topics is not None:
        state.generated_reply_message = generate_message(
            state.context,
            state.classified_category,
            state.selected_topics,
            dry_run=False,
        )

        state.step = "end: reply generated"
        return state

    # If the suggested topics are already generated, prompt the human to select topics
    if state.suggested_topics is not None:
        state.step = "next: select topics"
        return state

    # If the conversation is not classified, classify it
    if state.classified_category is None:
        state.classified_category = classify_conversation(
            state.context, CATEGORIES, dry_run=False
        )

    # Get the extended category with post-processing indicators
    extended_category = EXTENDED_CATEGORY_LOOKUP[state.classified_category.category]

    # If no reply is needed, return the current state as final state
    if not extended_category.reply_needed:
        state.step = "end: no reply needed"
        return state

    # If human action is required, run the actions summarizer and prompt the human to take action
    if extended_category.human_action_required:
        # TODO: Add actions summarizer
        state.step = "todo: human action required"
        return state

    # If human action is not required, suggest topics
    else:
        # Suggest topics
        state.suggested_topics = suggest_topics(
            state.context, state.classified_category, dry_run=False
        )

        state.step = "next: select topics"
        return state
