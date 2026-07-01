from __future__ import annotations


def hateful_memes_yes_no_prompt(meme_text: str) -> str:
    meme_text = (meme_text or "").strip()
    return (
        "You are judging a meme for hateful content. "
        "Use the image and the meme text together. "
        f"Meme text: {meme_text if meme_text else '[EMPTY]'}. "
        "Question: Is this meme hateful? "
        "Answer with exactly one token: Yes or No."
    )


def nlvr2_yes_no_prompt(statement: str) -> str:
    statement = (statement or "").strip()
    return (
        "You are given two images and a natural language statement about them. "
        "Use both images together to decide whether the statement is true. "
        f"Statement: {statement if statement else '[EMPTY]'}. "
        "Question: Is the statement correct for the two images? "
        "Answer with exactly one token: Yes or No."
    )


def scienceqa_multichoice_prompt(question: str, choices, option_labels, hint: str = "") -> str:
    question = (question or "").strip()
    hint = (hint or "").strip()
    rendered_choices = []
    for label, choice in zip(option_labels, choices):
        rendered_choices.append(f"{label}. {str(choice).strip()}")

    parts = [
        "You are solving a science multiple-choice question.",
        f"Question: {question if question else '[EMPTY]'}",
    ]
    if hint:
        parts.append(f"Context: {hint}")
    parts.append("Options:")
    parts.extend(rendered_choices)
    parts.append(f"Answer with exactly one option letter: {', '.join(option_labels)}.")
    return " ".join(parts)


def mmimdb_multilabel_prompt(title: str, plot: str, label_names) -> str:
    title = (title or "").strip()
    plot = (plot or "").strip()
    rendered_labels = ", ".join([str(name).strip() for name in label_names if str(name).strip()])
    parts = [
        "You are predicting all applicable movie genres from a poster and a plot synopsis.",
        f"Title: {title if title else '[EMPTY]'}",
        f"Plot: {plot if plot else '[EMPTY]'}",
    ]
    if rendered_labels:
        parts.append(f"Candidate genres: {rendered_labels}.")
    parts.append("Use the image and text together to determine every genre that applies.")
    return " ".join(parts)
