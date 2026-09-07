"""Explicit dataset conversion and pinned official scorer entrypoints."""


def vsi_row(raw):
    if not raw.get("media") and not raw.get("frame_paths"):
        raise ValueError("Attach decoded ordered frame_paths to each VSI source row before converting")
    task = raw["question_type"]
    numeric = task in {"object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"}
    return {**raw, "dataset":"VSI-Bench", "id":raw["id"], "scene_id":raw["dataset"]+"/"+raw["scene_name"],
            "question":raw["question"], "answer":raw["ground_truth"],
            "choices":None if numeric else raw["options"], "task":task,
            "metric":"numeric" if numeric else "choice"}


def vsi_official(rows, predictions, config):
    from .vendor.vsi.utils import vsibench_process_results, vsibench_aggregate_results, MCA_QUESTION_TYPES, NA_QUESTION_TYPES
    from .data import key
    expected = set(MCA_QUESTION_TYPES + NA_QUESTION_TYPES)
    if {r["task"] for r in rows} != expected:
        raise ValueError("Pinned VSI overall requires all 10 original types (8 aggregate tasks); use generic scores for a subset")
    by_id = {p["id"]:p for p in predictions}
    results = []
    for row in rows:
        doc = {"id":row["id"],"question_type":row["task"],"ground_truth":row["answer"]}
        # Deliberately preserve the upstream answer parser and threshold arithmetic.
        # Use direct-answer prompts for this scorer, not XML-formatted responses.
        response = by_id.get(key(row), {}).get("response", "")
        results.append(vsibench_process_results(doc, [response])["vsibench_score"])
    return {"source_commit":"51e089c3ae69b9435e9489058610f5b3964c56a8",
            "status":"official_scoring_code_with_declared_input_protocol",
            "metrics":dict(vsibench_aggregate_results(results))}


def conversation_qa(raw):
    """Single-turn LLaVA/SPAR-style conversations; reject multi-turn ambiguity."""
    conversations = raw.get("conversations", [])
    users = [v["value"] for v in conversations if v.get("from") in {"human", "user"}]
    assistants = [v["value"] for v in conversations if v.get("from") in {"gpt", "assistant"}]
    if len(users) != 1 or len(assistants) != 1:
        raise ValueError("Converter requires exactly one user and one assistant turn")
    return {**raw, "question":users[0].replace("<image>", "").replace("<video>", "").strip(),
            "answer":assistants[0], "media":raw.get("frame_paths",raw.get("images",raw.get("image",[])))}
