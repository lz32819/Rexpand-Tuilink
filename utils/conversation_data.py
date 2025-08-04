import json
from rexpand_pyutils_file import read_file

from models.context import ConversationMessage


def get_attachments_text(render_content):
    try:
        attachments = json.loads(render_content)
        return ", ".join(
            [
                f'[{attachment["file"]["name"]}]'
                for attachment in attachments
                if attachment["file"]
            ]
        )
    except Exception as e:
        print(e)
        return ""


def load_conversation_data(
    file_path: str, excel_sheet_name: str = "Result 1"
) -> list[dict]:
    raw_conversation_data = read_file(file_path, excel_sheet_name=excel_sheet_name)

    grouped_conversation_data: dict[str, list[ConversationMessage]] = {}
    for row in raw_conversation_data:
        if row["message_body_render_format"] == "RECALLED":
            continue

        key = row["user_id"] + "|" + row["conversation_id"]
        if key not in grouped_conversation_data:
            grouped_conversation_data[key] = []

        grouped_conversation_data[key].append(
            ConversationMessage(
                id=row["id"],
                role="job seeker" if row["sender_name"] == "You" else "referrer",
                name=(
                    row["user_name"]
                    if row["sender_name"] == "You"
                    else row["sender_name"]
                ),
                body_text=row["body_text"],
                attachments_text=get_attachments_text(row["render_content"]),
                delivered_at=row["delivered_at"],
            )
        )

    return [
        sorted(values, key=lambda x: x.delivered_at)
        for values in grouped_conversation_data.values()
    ]
