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
from utils.question_detector import detect_questions
from utils.action_completion_detector import detect_completed_actions, are_all_actions_completed


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
        # First check if inference result exists regarding whether referral is still possible
        if state["context"].referral_possibility is not None:
            # Inference result exists, go to topic suggester with actions
            return "suggest_topics_with_actions"
        
        # If inference result does not exist, see if there are existing questions
        if state["context"].questions_exist is None:
            # Detect questions first
            question_result = detect_questions(state["context"].context, dry_run=False)
            state["context"].questions_exist = question_result["questions_exist"]
            state["context"].questions_answered = question_result["questions_answered"]
        
        # Check if questions exist
        if not state["context"].questions_exist:
            # No questions exist, check if actions summary exists
            if state["context"].actions_summary is None:
                # Generate actions summary for human review
                return "generate_actions_summary"
            else:
                # Actions summary exists, check if actions are completed
                return "check_actions_completion"
        
        # Questions exist, check if they have all been answered
        if not state["context"].questions_answered:
            # Questions exist but not all answered, prompt user to answer questions
            return "prompt_answer_questions"
        
        # Questions exist and all have been answered, go to inferencer
        return "infer_referral_possibility"

    # If human action is not required, suggest topics without actions
    else:
        return "suggest_topics_no_action"


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


def suggest_topics_with_actions(state: WorkflowState) -> WorkflowState:
    """Suggest topics when human action is required and referral possibility is assessed."""
    state["context"].suggested_topics = suggest_topics(
        state["context"].context, 
        state["context"].classified_category, 
        state["context"].referral_possibility,
        dry_run=False
    )
    state["context"].step = "next: select topics"
    return state


def generate_actions_summary(state: WorkflowState) -> WorkflowState:
    """Generate actions summary for human review when no questions exist."""
    state["context"].actions_summary = summarize_actions(
        state["context"].context, 
        state["context"].classified_category, 
        dry_run=False
    )
    state["context"].step = "next: human action required"
    return state


def prompt_answer_questions(state: WorkflowState) -> WorkflowState:
    """Prompt user to answer questions."""
    state["context"].step = "next: answer questions"
    return state


def infer_referral_possibility_node(state: WorkflowState) -> WorkflowState:
    """Run referral possibility inference after questions are answered."""
    state["context"].referral_possibility = infer_referral_possibility(
        state["context"].context, 
        state["context"].classified_category, 
        state["context"].actions_summary, 
        dry_run=False
    )
    state["context"].step = "next: referral possibility assessed"
    return state


def check_actions_completion(state: WorkflowState) -> WorkflowState:
    """Check if actions have been completed and update the state accordingly."""
    # Detect newly completed actions
    newly_completed_actions = detect_completed_actions(
        state["context"].context,
        state["context"].actions_summary,
        state["context"].completed_actions,
        dry_run=False
    )
    
    # Add newly completed actions to the state
    state["context"].completed_actions.extend(newly_completed_actions)
    
    # Check if all actions are completed
    all_actions_completed = are_all_actions_completed(
        state["context"].actions_summary,
        state["context"].completed_actions
    )
    
    if all_actions_completed:
        # All actions completed, proceed to referral possibility inference
        state["context"].step = "next: all actions completed"
        return state
    else:
        # Actions still pending, prompt user to complete them
        state["context"].step = "next: actions pending"
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
    workflow.add_node("suggest_topics_no_action", suggest_topics_no_human_action)
    workflow.add_node("suggest_topics_with_actions", suggest_topics_with_actions)
    workflow.add_node("generate_actions_summary", generate_actions_summary)
    workflow.add_node("prompt_answer_questions", prompt_answer_questions)
    workflow.add_node("check_actions_completion", check_actions_completion)
    workflow.add_node("infer_referral_possibility", infer_referral_possibility_node)
    
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
        {
            "suggest_topics_no_action": "suggest_topics_no_action",
            "suggest_topics_with_actions": "suggest_topics_with_actions", 
            "generate_actions_summary": "generate_actions_summary",
            "prompt_answer_questions": "prompt_answer_questions",
            "check_actions_completion": "check_actions_completion",
            "infer_referral_possibility": "infer_referral_possibility"
        }
    )
    
    # Add conditional edges for check_actions_completion
    workflow.add_conditional_edges(
        "check_actions_completion",
        lambda state: "infer_referral_possibility" if state["context"].step == "next: all actions completed" else "prompt_complete_actions",
        {
            "infer_referral_possibility": "infer_referral_possibility",
            "prompt_complete_actions": "prompt_answer_questions"
        }
    )
    
    # Add edges to END for terminal nodes
    workflow.add_edge("suggest_topics_no_action", END)
    workflow.add_edge("suggest_topics_with_actions", END)
    workflow.add_edge("generate_actions_summary", END)
    workflow.add_edge("prompt_answer_questions", END)
    workflow.add_edge("infer_referral_possibility", END)
    
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
