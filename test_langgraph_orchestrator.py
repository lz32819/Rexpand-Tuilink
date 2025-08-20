#!/usr/bin/env python3
"""
Test script for the new LangGraph-based orchestrator.
This script demonstrates how the new workflow structure works.
"""

import logging
logging.basicConfig(level=logging.INFO)

from utils.conversation_data import load_conversation_data
from models.context import Context
from models.workflow import State
from nodes.orchestrator import orchestrate, create_workflow


def test_langgraph_orchestrator():
    """Test the new LangGraph orchestrator functionality."""
    print("Testing LangGraph orchestrator functionality...")
    
    # Load conversations
    conversations = load_conversation_data('./input/convo_2454_rows.xlsx')
    
    # Test workflow creation
    print("\n1. Testing workflow creation...")
    workflow = create_workflow()
    print(f"   Workflow created successfully: {type(workflow)}")
    print(f"   Workflow nodes: {list(workflow.nodes.keys())}")
    
    # Test with a simple conversation
    print("\n2. Testing orchestrator with conversation...")
    index = 10
    messages_end = 5
    messages = conversations[index][:messages_end]
    context = Context(messages=messages)
    
    # Create state and run orchestrator
    state = State(context=context)
    print(f"   Initial state step: {state.step}")
    print(f"   Initial classified category: {state.classified_category}")
    
    # Run the orchestrator
    try:
        result_state = orchestrate(state)
        print(f"   Final state step: {result_state.step}")
        print(f"   Final classified category: {result_state.classified_category}")
        print(f"   Suggested topics: {result_state.suggested_topics}")
        print(f"   Generated reply: {result_state.generated_reply_message}")
        
        print("\n✅ LangGraph orchestrator test completed successfully!")
        
    except Exception as e:
        print(f"   ❌ Error during orchestration: {e}")
        import traceback
        traceback.print_exc()


def test_workflow_structure():
    """Test the workflow structure and visualization."""
    print("\n3. Testing workflow structure...")
    
    try:
        workflow = create_workflow()
        
        # Print workflow information
        print(f"   Workflow type: {type(workflow)}")
        print(f"   Nodes: {list(workflow.nodes.keys())}")
        
        # Try to get the workflow graph structure
        print(f"   Workflow compiled successfully: {workflow.compile() is not None}")
        
        print("   ✅ Workflow structure test completed!")
        
    except Exception as e:
        print(f"   ❌ Error testing workflow structure: {e}")


if __name__ == "__main__":
    test_langgraph_orchestrator()
    test_workflow_structure()
