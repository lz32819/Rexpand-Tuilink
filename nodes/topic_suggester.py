import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import ClassifierResult, TopicSuggesterResult, ReferralPossibilityResult
from utils.llm import invoke_llm


def suggest_topics(
    context: Context, 
    classified_category: ClassifierResult, 
    referral_possibility: ReferralPossibilityResult | None = None,
    dry_run: bool = False
) -> TopicSuggesterResult:
    system_prompt = f"""\
You are a helpful assistant that suggests topics for the job seeker to reply for a potential referral or intro call.
You will be given the conversation history, classified category, job seeker profile (optional), referrer profile (optional), and referral possibility assessment (optional).

When suggesting topics, always evaluate the latest messages first and consider the referral possibility assessment.
Never make up facts.

Topic Examples (but not limited to):
1. Thank you
2. Express interest
3. Express background match
4. Ask for a brief call
5. Ask for alternative referrers
6. Follow-up
7. Apologize and ask for another chance (if referral was lost)
8. Ask for feedback on how to improve
9. Suggest alternative ways to connect
10. Thank for the opportunity and move on gracefully (if referral is not possible)
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}
"""

    if referral_possibility:
        user_prompt += f"""
Referral Possibility Assessment:
- Referral Possible: {referral_possibility.referral_possible}
- Confidence: {referral_possibility.confidence}
- Reason: {referral_possibility.reason}
- Next Steps: {referral_possibility.next_steps}
- Barriers: {referral_possibility.barriers}

Based on this assessment, suggest topics that are appropriate for the current situation.
If referral is not possible, focus on graceful exit strategies and learning opportunities.
If referral is still possible, focus on actions that can secure it.
"""

    if context.user_profile:
        user_prompt += f"""
Job Seeker Profile:
{context.user_profile}
"""

    if context.referrer_profile:
        user_prompt += f"""
Referrer Profile:
{context.referrer_profile}
"""

    output_schema = {
        "name": "topic_suggester_result",
        "strict": True,
        "type": "json_schema",
        "schema": TopicSuggesterResult.model_json_schema(),
    }

    if dry_run:
        return system_prompt, user_prompt, output_schema

    response = invoke_llm(
        input=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        text={"format": output_schema},
        use_cache=True,
    )

    return TopicSuggesterResult(**json.loads(response.content[0]["text"]))
