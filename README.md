# Tuilink Project - AI Auto Reply

An intelligent system for generating contextually appropriate replies in professional conversations, particularly focused on job referral scenarios.

## Overview

This project implements an automated reply generation system that:

1. Analyzes conversation context
2. Classifies conversation categories
3. Suggests relevant topics for replies
4. Generates professional and contextually appropriate responses

## Project Structure

```
.
├── input/                  # Input data directory
│   ├── categories.json     # Category definitions
│   └── convo_2454_rows.xlsx # Conversation dataset
├── models/                 # Core data models
├── nodes/                  # Processing nodes
├── output/                 # Generated outputs
├── utils/                  # Utility functions
└── run.ipynb              # Main execution notebook
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
