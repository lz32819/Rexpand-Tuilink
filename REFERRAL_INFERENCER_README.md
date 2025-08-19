# Referral Inferencer

## Overview

The Referral Inferencer is a new component that evaluates whether it's still possible to get a referral after the user has taken specific actions. This component bridges the gap between the actions summarizer and topic suggester, providing intelligent assessment of referral possibilities.

## Architecture

### New Components

1. **ReferralInferencer Node** (`nodes/referral_inferencer.py`)
   - Analyzes conversation context and user actions
   - Determines referral possibility with confidence scoring
   - Provides next steps and identifies barriers

2. **ReferralPossibilityResult Model** (`models/llm_result.py`)
   - `referral_possible`: Boolean indicating if referral is still achievable
   - `confidence`: Confidence level in the assessment
   - `reason`: Detailed reasoning for the assessment
   - `next_steps`: List of actionable steps to secure referral
   - `barriers`: List of obstacles that might prevent referral

### Updated Components

1. **State Model** (`models/workflow.py`)
   - Added `referral_possibility` field to track inference results

2. **Orchestrator** (`nodes/orchestrator.py`)
   - Automatically runs referral inference after actions are taken
   - Routes to topic suggestion based on referral possibility assessment

3. **Topic Suggester** (`nodes/topic_suggester.py`)
   - Now considers referral possibility when suggesting topics
   - Provides contextually appropriate suggestions based on referral status

## Workflow

```
User Action Required → Actions Summary → Referral Inference → Topic Suggestion
```

1. **Actions Summary**: System identifies required human actions
2. **User Takes Action**: Human performs the required actions
3. **Referral Inference**: System automatically assesses referral possibility
4. **Topic Suggestion**: System suggests appropriate topics based on referral status

## Usage

### Running the Referral Inferencer

```python
from nodes.referral_inferencer import infer_referral_possibility

# After user actions are taken
referral_result = infer_referral_possibility(
    context=context,
    classified_category=classified_category,
    actions_summary=actions_summary,
    dry_run=False
)

# Check referral possibility
if referral_result.referral_possible:
    print("Referral is still possible!")
    print("Next steps:", referral_result.next_steps)
else:
    print("Referral may not be possible")
    print("Barriers:", referral_result.barriers)
```

### Testing

Run the test script to see the referral inferencer in action:

```bash
cd Rexpand-Tuilink
python test_referral_inferencer.py
```

## Benefits

1. **Intelligent Assessment**: Automatically evaluates referral possibilities after user actions
2. **Contextual Topic Suggestions**: Topics are tailored based on referral status
3. **Actionable Guidance**: Provides specific next steps and identifies barriers
4. **Seamless Integration**: Works automatically within the existing workflow
5. **Real-time Adaptation**: Adjusts strategy based on current conversation state

## Example Output

```
Referral Possibility Result:
- Referral Possible: True
- Confidence: 0.85
- Reason: The user's follow-up message was professional and timely, showing continued interest without being pushy.
- Next Steps: ['Send a brief thank you', 'Ask for a 15-minute call', 'Share relevant portfolio work']
- Barriers: ['Referrer may be busy', 'Timing of follow-up']

Topics with Referral Assessment:
- Ask for a brief call (confidence: 0.95)
  Reason: The referral is still possible, so requesting a call is appropriate
- Share portfolio work (confidence: 0.85)
  Reason: Demonstrating value can help secure the referral
```

## Configuration

The referral inferencer uses the same LLM configuration as other components. The system prompt is designed to be realistic and honest in assessments, avoiding overly optimistic predictions.
