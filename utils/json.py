from pydantic import BaseModel


def to_json_compatible(obj):
    """
    Recursively convert supported objects to JSON-friendly Python values.

    We avoid importing heavy optional dependencies here so the Lambda runtime
    can stay small. NumPy objects are handled by duck-typing when present.
    """
    module_name = type(obj).__module__
    if module_name.startswith("numpy") and hasattr(obj, "tolist"):
        return obj.tolist()
    elif module_name.startswith("numpy") and hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_compatible(v) for v in obj]
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    else:
        return obj
