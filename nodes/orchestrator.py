from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from rexpand_pyutils_file import read_file

from models.category import Category, ExtendedCategory
from models.workflow import State
from nodes.actions_summarizer import summarize_actions
from nodes.classifier import classify_conversation
from nodes.message_generator import generate_message
from nodes.referral_inferencer import infer_referral_possibility
from nodes.topic_suggester import suggest_topics
from utils.action_completion_detector import are_all_actions_completed


CATEGORIES = [Category(**category) for category in read_file("./input/categories.json")]
EXTENDED_CATEGORY_LOOKUP: dict[str, ExtendedCategory] = {
    category["category"]: ExtendedCategory(**category)
    for category in read_file("./input/categories.json")
}
# Group-level fallback for cases where the classifier returns a category_group instead of a category.
# (We assume flags are consistent within a group; if not, we keep the first seen.)
EXTENDED_CATEGORY_GROUP_LOOKUP: dict[str, ExtendedCategory] = {}
for _cat, _ext in EXTENDED_CATEGORY_LOOKUP.items():
    EXTENDED_CATEGORY_GROUP_LOOKUP.setdefault(_ext.category_group, _ext)


class WorkflowState(TypedDict):
    """Wrapper state for the LangGraph workflow."""

    context: State


def _extended_category(state: WorkflowState) -> ExtendedCategory:
    return EXTENDED_CATEGORY_LOOKUP[state["context"].classified_category.category]


def _fingerprint(context: State) -> str:
    """
    Fingerprint the current conversation so we can reset cached routing fields when
    new messages arrive (new conversation iteration).
    """
    messages = context.context.messages
    if not messages:
        return "empty"
    last = messages[-1]
    return f"{len(messages)}:{last.id}:{last.delivered_at}"


def _reset_cached_fields(s: State) -> None:
    """
    Reset cached node outputs/routing decisions so a new conversation iteration can run cleanly.

    IMPORTANT: We intentionally do NOT clear `generated_reply_message` here, so callers can still
    read the last generated message after an iteration ends.
    """
    s.step = None
    s.classified_category = None
    s.reply_needed = None
    s.human_action_required = None
    s.actions_fulfilled = None
    s.suggested_topics = None
    s.selected_topics = None
    s.actions_summary = None
    s.referral_possibility = None
    s.questions_exist = None
    s.questions_answered = None
    s.completed_actions = []


# -----------------------------
# Classifier-first entry node
# -----------------------------
def classify_and_cache_node(state: WorkflowState) -> WorkflowState:
    s = state["context"]

    fp = _fingerprint(state["context"])
    if s.context_fingerprint != fp:
        _reset_cached_fields(s)
        s.context_fingerprint = fp

    if s.classified_category is None:
        s.classified_category = classify_conversation(s.context, CATEGORIES, dry_run=False)

    ext = EXTENDED_CATEGORY_LOOKUP.get(s.classified_category.category)
    if ext is None:
        ext = EXTENDED_CATEGORY_GROUP_LOOKUP.get(s.classified_category.category)
    # if ext is None:
    #     # Unknown label: default to a conservative flow that still helps the user.
    #     # (Reply needed = True; no human action required).
    #     if s.reply_needed is None:
    #         s.reply_needed = True
    #     if s.human_action_required is None:
    #         s.human_action_required = False
    #     s.step = "warn: unknown category label"
    #     return state
    # if s.reply_needed is None:
    #     s.reply_needed = ext.reply_needed
    # if s.human_action_required is None:
    #     s.human_action_required = ext.human_action_required

    return state


def reply_needed_router(state: WorkflowState) -> str:
    return "end" if state["context"].reply_needed is False else "next"


def no_reply_end_node(state: WorkflowState) -> WorkflowState:
    state["context"].step = "end: no reply needed"
    return state


def action_required_router(state: WorkflowState) -> str:
    return "actions" if state["context"].human_action_required else "no_actions"


# -----------------------------
# No-human-action path
# -----------------------------
def topic_selection_router(state: WorkflowState) -> str:
    return "generate" if state["context"].selected_topics is not None else "suggest"


def suggest_topics_node(state: WorkflowState) -> WorkflowState:
    s = state["context"]
    if s.suggested_topics is None:
        s.suggested_topics = suggest_topics(
            s.context,
            s.classified_category,
            referral_possibility=s.referral_possibility,
            dry_run=False,
        )
    s.step = "next: select topics"
    return state


def generate_message_node(state: WorkflowState) -> WorkflowState:
    s = state["context"]
    s.generated_reply_message = generate_message(
        s.context,
        s.classified_category,
        s.selected_topics,
        dry_run=False,
    )
    s.step = "end: reply generated"
    return state


# -----------------------------
# Human-action-required path
# -----------------------------
def ensure_actions_summary_node(state: WorkflowState) -> WorkflowState:
    s = state["context"]
    if s.actions_summary is None:
        s.actions_summary = summarize_actions(s.context, s.classified_category, dry_run=False)
    return state


def actions_fulfilled_node(state: WorkflowState) -> WorkflowState:
    s = state["context"]
    if s.actions_fulfilled is None:
        s.actions_fulfilled = are_all_actions_completed(s.actions_summary, s.completed_actions)
    return state


def actions_fulfilled_router(state: WorkflowState) -> str:
    return "fulfilled" if state["context"].actions_fulfilled else "unfulfilled"


def prompt_fulfill_actions_node(state: WorkflowState) -> WorkflowState:
    state["context"].step = "next: fulfill actions"
    return state


def infer_referral_possibility_node(state: WorkflowState) -> WorkflowState:
    s = state["context"]
    if s.referral_possibility is None:
        s.referral_possibility = infer_referral_possibility(
            s.context,
            s.classified_category,
            s.actions_summary,
            dry_run=False,
        )
    return state


def create_workflow() -> StateGraph:
    """
    LangGraph Orchestrator (AI helper for job seekers to communicate with referrers).

    Control-flow (updated):
    - Entry: classifier (cache classification + routing decisions)
    - If reply not needed -> end
    - If no human action required -> suggest topics -> user selects -> generate message -> end
    - If human action required -> summarize actions -> check fulfilled
        - If not fulfilled -> prompt user to fulfill actions -> end (current iteration)
        - If fulfilled -> infer referral possibility -> suggest topics -> user selects -> generate message -> end
    """

    workflow = StateGraph(WorkflowState)

    workflow.add_node("classify_and_cache", classify_and_cache_node)
    workflow.add_node("no_reply_end", no_reply_end_node)

    workflow.add_node("check_action_required", lambda s: s)
    workflow.add_node("suggest_topics", suggest_topics_node)
    workflow.add_node("generate_message", generate_message_node)

    workflow.add_node("ensure_actions_summary", ensure_actions_summary_node)
    workflow.add_node("actions_fulfilled", actions_fulfilled_node)
    workflow.add_node("prompt_fulfill_actions", prompt_fulfill_actions_node)
    workflow.add_node("infer_referral_possibility", infer_referral_possibility_node)

    workflow.set_entry_point("classify_and_cache")

    # After classification, decide whether a reply is needed.
    workflow.add_conditional_edges(
        "classify_and_cache",
        reply_needed_router,
        {"end": "no_reply_end", "next": "check_action_required"},
    )
    workflow.add_edge("no_reply_end", END)

    # If reply is needed, decide whether human action is required.
    workflow.add_conditional_edges(
        "check_action_required",
        action_required_router,
        {"no_actions": "topic_selection_gate", "actions": "ensure_actions_summary"},
    )

    # No-action flow: if topics already selected -> generate; else suggest topics.
    workflow.add_node("topic_selection_gate", lambda s: s)
    workflow.add_conditional_edges(
        "topic_selection_gate",
        topic_selection_router,
        {"generate": "generate_message", "suggest": "suggest_topics"},
    )
    workflow.add_edge("suggest_topics", END)
    workflow.add_edge("generate_message", END)

    # Action-required flow: summarize actions -> check fulfilled.
    workflow.add_edge("ensure_actions_summary", "actions_fulfilled")
    workflow.add_conditional_edges(
        "actions_fulfilled",
        actions_fulfilled_router,
        {"unfulfilled": "prompt_fulfill_actions", "fulfilled": "infer_referral_possibility"},
    )
    workflow.add_edge("prompt_fulfill_actions", END)

    # If fulfilled, infer referral possibility then proceed to topic selection/generation.
    workflow.add_edge("infer_referral_possibility", "topic_selection_gate")

    return workflow


def orchestrate(state: State) -> State:
    """Run the LangGraph orchestrator and return the updated workflow State."""
    app = create_workflow().compile()
    result = app.invoke(WorkflowState(context=state))
    return result["context"]
