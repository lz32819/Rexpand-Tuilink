from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from rexpand_pyutils_file import read_file
from models.category import Category, ExtendedCategory
from models.workflow import State
from nodes.classifier import classify_conversation
from nodes.message_generator import generate_message
from nodes.topic_suggester import suggest_topics
from nodes.actions_summarizer import summarize_actions
from nodes.referral_inferencer import infer_referral_possibility


CATEGORIES = [Category(**category) for category in read_file("./input/categories.json")]
EXTENDED_CATEGORY_LOOKUP = {
    category["category"]: ExtendedCategory(**category)
    for category in read_file("./input/categories.json")
}


class WorkflowState(TypedDict):
    """State for the LangGraph workflow."""
    context: State
    step: str


def check_reply_generated_node(state: WorkflowState) -> WorkflowState:
    """Node that checks if reply message is already generated."""
    if state["context"].generated_reply_message is not None:
        state["context"].step = "end: reply generated"
    return state


def check_topics_selected_node(state: WorkflowState) -> WorkflowState:
    """Node that checks if topics are already selected."""
    if state["context"].selected_topics is not None:
        # Generate reply message
        state["context"].generated_reply_message = generate_message(
            state["context"].context,
            state["context"].classified_category,
            state["context"].selected_topics,
            dry_run=False,
        )
        state["context"].step = "end: reply generated"
    return state


def check_topics_suggested_node(state: WorkflowState) -> WorkflowState:
    """Node that checks if topics are already suggested."""
    if state["context"].suggested_topics is not None:
        state["context"].step = "next: select topics"
    return state


def classify_conversation_node(state: WorkflowState) -> WorkflowState:
    """Classify the conversation if not already classified."""
    if state["context"].classified_category is None:
        state["context"].classified_category = classify_conversation(
            state["context"].context, CATEGORIES, dry_run=False
        )
    return state


def check_reply_needed_node(state: WorkflowState) -> WorkflowState:
    """Node that checks if a reply is needed."""
    extended_category = EXTENDED_CATEGORY_LOOKUP[state["context"].classified_category.category]
    if not extended_category.reply_needed:
        state["context"].step = "end: no reply needed"
    return state


def check_human_action_required_node(state: WorkflowState) -> WorkflowState:
    """Node that checks if human action is required."""
    # This node just updates the state, routing is handled by conditional edges
    return state


# Conditional edge functions that handle routing
def check_reply_generated_router(state: WorkflowState) -> str:
    """Router for check_reply_generated node."""
    if state["context"].generated_reply_message is not None:
        return "end"
    return "next"


def check_topics_selected_router(state: WorkflowState) -> str:
    """Router for check_topics_selected node."""
    if state["context"].selected_topics is not None:
        return "end"
    return "next"


def check_topics_suggested_router(state: WorkflowState) -> str:
    """Router for check_topics_suggested node."""
    if state["context"].suggested_topics is not None:
        return "end"
    return "next"


def check_reply_needed_router(state: WorkflowState) -> str:
    """Router for check_reply_needed node."""
    extended_category = EXTENDED_CATEGORY_LOOKUP[state["context"].classified_category.category]
    if not extended_category.reply_needed:
        return "end"
    return "next"


def check_human_action_required_router(state: WorkflowState) -> str:
    """Router for check_human_action_required node."""
    extended_category = EXTENDED_CATEGORY_LOOKUP[state["context"].classified_category.category]
    if extended_category.human_action_required:
        return "human_action"
    else:
        return "no_human_action"


def handle_human_action_workflow(state: WorkflowState) -> str:
    """Handle the human action workflow."""
    # Check if actions summary exists but referral possibility hasn't been assessed yet
    if (state["context"].actions_summary is not None and 
        state["context"].referral_possibility is None):
        # Run referral possibility inference after actions were taken
        state["context"].referral_possibility = infer_referral_possibility(
            state["context"].context, 
            state["context"].classified_category, 
            state["context"].actions_summary, 
            dry_run=False
        )
        state["context"].step = "next: referral possibility assessed"
        return "end"
    
    # Check if referral possibility has been assessed
    if state["context"].referral_possibility is not None:
        # Suggest topics based on referral possibility
        state["context"].suggested_topics = suggest_topics(
            state["context"].context, 
            state["context"].classified_category, 
            state["context"].referral_possibility,
            dry_run=False
        )
        state["context"].step = "next: select topics"
        return "end"
    
    # Generate actions summary for human review
    state["context"].actions_summary = summarize_actions(
        state["context"].context, 
        state["context"].classified_category, 
        dry_run=False
    )
    state["context"].step = "next: human action required"
    return "end"


def suggest_topics_no_human_action(state: WorkflowState) -> WorkflowState:
    """Suggest topics when no human action is required."""
    state["context"].suggested_topics = suggest_topics(
        state["context"].context, 
        state["context"].classified_category, 
        referral_possibility=None, 
        dry_run=False
    )
    state["context"].step = "next: select topics"
    return state


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow."""
    workflow = StateGraph(WorkflowState)
    
    # Add all nodes
    workflow.add_node("start", lambda state: state)  # Entry point node
    workflow.add_node("check_reply_generated", check_reply_generated_node)
    workflow.add_node("check_topics_selected", check_topics_selected_node)
    workflow.add_node("check_topics_suggested", check_topics_suggested_node)
    workflow.add_node("classify_conversation", classify_conversation_node)
    workflow.add_node("check_reply_needed", check_reply_needed_node)
    workflow.add_node("check_human_action_required", check_human_action_required_node)
    workflow.add_node("handle_human_action", lambda state: handle_human_action_workflow(state))
    workflow.add_node("suggest_topics_no_human_action", suggest_topics_no_human_action)
    
    # Add conditional edges with path_map
    workflow.add_conditional_edges(
        "start", 
        check_reply_generated_router,
        {"end": END, "next": "check_topics_selected"}
    )
    workflow.add_conditional_edges(
        "check_topics_selected", 
        check_topics_selected_router,
        {"end": END, "next": "check_topics_suggested"}
    )
    workflow.add_conditional_edges(
        "check_topics_suggested", 
        check_topics_suggested_router,
        {"end": END, "next": "classify_conversation"}
    )
    workflow.add_conditional_edges(
        "classify_conversation", 
        check_reply_needed_router,
        {"end": END, "next": "check_human_action_required"}
    )
    workflow.add_conditional_edges(
        "check_human_action_required", 
        check_human_action_required_router,
        {"human_action": "handle_human_action", "no_human_action": "suggest_topics_no_human_action"}
    )
    
    # Add conditional edges for handle_human_action
    workflow.add_conditional_edges(
        "handle_human_action",
        lambda state: "end",
        {"end": END}
    )
    
    workflow.add_edge("suggest_topics_no_human_action", END)
    
    # Set entry point
    workflow.set_entry_point("start")
    
    return workflow


def orchestrate(state: State) -> State:
    """Orchestrate the workflow using LangGraph."""
    # Create workflow
    app = create_workflow().compile()
    
    # Create initial workflow state
    workflow_state = WorkflowState(
        context=state,
        step="start"
    )
    
    # Run the workflow
    result = app.invoke(workflow_state)
    
    # Return the updated state
    return result["context"]
