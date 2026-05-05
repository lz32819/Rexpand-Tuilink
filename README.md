# Tuilink Project - AI Auto Reply

An intelligent system for generating contextually appropriate replies in professional conversations, particularly focused on job referral scenarios.

## Overview

This project implements an automated reply generation system that:

1. Analyzes conversation context
2. Classifies conversation categories
3. Suggests relevant topics for replies and auto-selects them in the API flow
4. Generates professional and contextually appropriate responses

## Project Structure

```
.
├── ai_auto_reply.py        # Deployment entrypoint
├── input/                  # Input data directory
│   ├── categories.json     # Category definitions
│   └── convo_2454_rows.xlsx # Local dataset
├── models/                 # Core data models
├── nodes/                  # Processing nodes
├── scripts/                # Helper scripts
├── utils/                  # Utility functions
└── run.ipynb               # Notebook for local exploration
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

The project uses environment variables for configuration. Create a `.env` file with the following necessary credentials and settings:

```bash
OPENAI_API_KEY=<your-openai-api-key>
```

## API Request Body

The deployment entrypoint accepts:

- `messages`: a list of conversation messages for a fresh run
- `state`: a previously returned `state` object to resume the workflow
- `actions_completed`: optional compatibility boolean; when `true`, marks
  `completion_boolean` actions in `state.actions_summary` as completed

Example Postman body for resuming after the API previously returned
`"step": "next: fulfill actions"`:

```json
{
  "state": {
    "...": "paste the previous response body.state here, then update each action",
    "actions_summary": {
      "actions": [
        {
          "action": "Answer the referrer's question about sponsorship.",
          "input_type": "message_response",
          "response_text": "I do not need sponsorship.",
          "completed": false
        },
        {
          "action": "Send your resume.",
          "input_type": "completion_boolean",
          "completed": true
        }
      ]
    }
  }
}
```

For each action:

- `input_type = "message_response"`: render a text entry and store the user's
  answer in `response_text`
- `input_type = "completion_boolean"`: render a boolean control and store the
  result in `completed`

When the updated `state` is submitted back to the API, text responses are added
to the conversation as new job seeker messages automatically. Once all actions
have either a completed boolean or a non-empty response text, the workflow moves
on to the next stage.

You can also include `messages` together with `state` if you want to replace
`state.context.messages` with an updated conversation history.

## Deployment Bundle

Build a clean folder with only the files needed for online deployment:

```bash
python scripts/build_deployment_bundle.py --force
```

This creates `deployment_bundle/` with the runtime source files, `input/categories.json`,
`requirements.txt`, `ai_auto_reply.py`, and a generated `.env.example`.

LLM cache files are written to `./.cache` during local runs when that path is writable.
In deployed environments with a read-only app directory, the runtime automatically falls
back to a writable system temp directory. You can override this with `LLM_CACHE_DIR`.
