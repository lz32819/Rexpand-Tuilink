#!/usr/bin/env python3
"""
Test script for the referral inferencer functionality.
This script demonstrates how the referral inferencer works after user actions.
"""

import logging
logging.basicConfig(level=logging.INFO)

from utils.conversation_data import load_conversation_data
from models.context import Context
from models.workflow import State
from nodes.orchestrator import orchestrate
from nodes.referral_inferencer import infer_referral_possibility
from nodes.topic_suggester import suggest_topics


def test_referral_inferencer():
    """Test the referral inferencer functionality."""
    print("Testing referral inferencer functionality...")
    
    # Load conversations
    conversations = load_conversation_data('./input/convo_2454_rows.xlsx')
    
    # Try different conversations to find one that requires human action
    for index in range(10, 20):
        print(f"\n--- Testing conversation {index} ---")
        
        messages_end = 8
        messages = conversations[index][:messages_end]
        context = Context(messages=messages)
        
        # Create state and run orchestrator
        state = State(context=context)
        state = orchestrate(state)
        
        print(f"Step: {state.step}")
        print(f"Category: {state.classified_category.category if state.classified_category else 'None'}")
        
        # If human action is required, test the referral inferencer
        if state.actions_summary:
            print(f"\nActions Summary: {state.actions_summary.summary}")
            
            # Test referral possibility inference
            referral_result = infer_referral_possibility(
                context, 
                state.classified_category, 
                state.actions_summary, 
                dry_run=False
            )
            
            print(f"\nReferral Possibility Result:")
            print(f"- Referral Possible: {referral_result.referral_possible}")
            print(f"- Confidence: {referral_result.confidence}")
            print(f"- Reason: {referral_result.reason}")
            print(f"- Next Steps: {referral_result.next_steps}")
            print(f"- Barriers: {referral_result.barriers}")
            
            # Now test topic suggestion with referral possibility
            topics_with_referral = suggest_topics(
                context, 
                state.classified_category, 
                referral_result, 
                dry_run=False
            )
            
            print(f"\nTopics with Referral Assessment:")
            for topic in topics_with_referral.topics:
                print(f"- {topic.topic} (confidence: {topic.confidence})")
                print(f"  Reason: {topic.reason}")
            
            # Found a conversation that requires human action, so we can stop testing
            break
        else:
            print("No actions summary generated - this conversation may not require human action")
    
    print("\n--- Referral inferencer test completed ---")


if __name__ == "__main__":
    test_referral_inferencer()
