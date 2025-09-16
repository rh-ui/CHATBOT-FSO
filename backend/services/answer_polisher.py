import re

INTRO_PATTERNS = [
    r"^en tant qu[’']?expert.*?\.\s*",           # Remove AI self-reference intro
    r"^je peux vous fournir.*?\.\s*",            # Remove "I can provide..." intros
    r"^\*\*?réponse\s*:?(\*\*)?",                # Remove bold "Réponse:" headers
]

def polish_answer(answer_text: str, all_documents: list) -> str:
    """
    Cleans and formats chatbot's answer:
    - Removes generic AI intros
    - Fixes truncated names like 'MA...'
    - Formats sources as numbered list
    - Ensures a consistent, concise style
    """
    if not answer_text or not isinstance(answer_text, str):
        return answer_text

    cleaned = answer_text.strip()

    # --- Step 1: Remove boilerplate intros ---
    for pattern in INTRO_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # --- Step 2: Detect truncated name patterns ---
    truncated_pattern = r"\b([A-Z][A-Z']{1,})\.\.\."
    matches = re.findall(truncated_pattern, cleaned)
    if matches:
        for match in matches:
            possible_full_name = find_full_name_in_docs(match, all_documents)
            if possible_full_name:
                cleaned = cleaned.replace(match + "...", possible_full_name)

    # --- Step 3: Format sources as numbered list ---
    cleaned = format_sources(cleaned)

    # --- Step 4: Clean spacing ---
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse multiple spaces
    cleaned = cleaned.replace(" ,", ",").replace(" .", ".")  # Fix misplaced spaces
    cleaned = cleaned.strip()

    # --- Step 5: Capitalize first letter ---
    if cleaned and not cleaned[0].isupper():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def find_full_name_in_docs(name_fragment: str, all_documents: list) -> str:
    """Finds full name in the provided documents."""
    fragment_upper = name_fragment.upper()
    for doc in all_documents:
        text_parts = []
        if isinstance(doc, dict):
            for k in ['answer', 'meta', 'titre', 'title', 'snippet']:
                if k in doc:
                    val = doc[k]
                    if isinstance(val, dict):
                        text_parts.extend(str(v) for v in val.values())
                    else:
                        text_parts.append(str(val))
        elif isinstance(doc, str):
            text_parts.append(doc)

        combined_text = " ".join(text_parts).upper()
        match = re.search(
            rf"\b{fragment_upper}[A-ZÉÈÂÊÎÔÛÄËÏÖÜÀÇ]+(?:\s+[A-Z][A-ZÉÈÂÊÎÔÛÄËÏÖÜÀÇ]+)*",
            combined_text
        )
        if match:
            return match.group(0).title()
    return None


def format_sources(text: str) -> str:
    """Extracts and reformats sources into a numbered list."""
    # Match "**Sources :**" and everything after
    match = re.search(r"(\*\*\s*Sources?\s*:?\s*\*\*)(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return text

    header, sources_block = match.groups()

    # Split sources by line or bullet
    sources = re.split(r"[\n\r]*[\*\-•]\s*", sources_block.strip())
    sources = [s.strip() for s in sources if s.strip()]

    # Create numbered format
    numbered_sources = "\n".join(f"{i+1}. {src}" for i, src in enumerate(sources))

    # Replace in text
    main_text = text[:match.start()].strip()
    return f"{main_text}\n\n**Sources :**\n{numbered_sources}"
