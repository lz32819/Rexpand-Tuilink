import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import ClassifierResult, ActionsSummaryResult, ReferralPossibilityResult
from utils.llm import invoke_llm


def infer_referral_possibility(
    context: Context,
    classified_category: ClassifierResult,
    actions_summary: ActionsSummaryResult,
    dry_run: bool = False,
) -> ReferralPossibilityResult:
    system_prompt = f"""\
You are a referral possibility analyzer that evaluates whether it's still possible to get a referral after the user has taken specific actions.

You will be given:
1. The conversation history
2. The classified conversation category
3. A summary of actions that were taken

Your task is to analyze whether a referral is still achievable and provide:
1. A clear yes/no answer on referral possibility
2. Your confidence level in this assessment

Only provide detailed reasoning when the referral is NOT possible:
- If referral_possible = True: Set reason to null
- If referral_possible = False: Provide detailed reason explaining why referral is not possible

You should consider if the user can satisfy the requirements proposed by the referrer. For example, if the referrer mentions that the company does not provide sponsorship for the position while the user needs sponsorship, it would be impossible to get a referral. 

Be realistic and honest in your assessment. Don't sugar-coat if the referral opportunity is lost.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}

Actions Taken Summary:
{actions_summary}

Please analyze whether a referral is still possible after these actions.
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
        "name": "referral_possibility_result",
        "strict": True,
        "type": "json_schema",
        "schema": ReferralPossibilityResult.model_json_schema(),
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

    return ReferralPossibilityResult(**json.loads(response.content[0]["text"]))
