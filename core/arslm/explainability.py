def explain(output):
    return {
        "decision_path": "abstracted",
        "confidence": output.get("confidence", None)
    }
