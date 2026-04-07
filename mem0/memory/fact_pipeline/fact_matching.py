import re

ADD_WITH_ATTR_SIMILARITY_THRESHOLD = 0.5
ADD_WITH_ATTR_RELATIONSHIP_PRIORITY = {
    "CONTRADICT": 5,
    "MORE_COMPLETE": 4,
    "IDENTICAL": 3,
    "PARAPHRASE": 2,
    "LESS_COMPLETE": 1,
}
TYPE_SPECIFIC_MIN_SCORE = {
    "summary": 0.72,
    "topic": 0.60,
    "statement": 0.64,
    "decision": 0.64,
    "knowledge": 0.64,
    "todo": 0.68,
    "issue": 0.68,
}
FACT_FILTER_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "with", "from", "that", "this", "those",
    "these", "into", "onto", "over", "under", "about", "after", "before", "during", "need",
    "needs", "will", "would", "could", "should", "have", "has", "had", "was", "were", "are",
    "is", "to", "of", "in", "on", "at", "by", "it", "we", "they", "he", "she", "i", "you",
}


def extract_fact_keywords(text):
    if not isinstance(text, str):
        return set()
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in FACT_FILTER_STOPWORDS}


def normalize_metadata_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [str(value).strip().lower()] if str(value).strip() else []


def normalize_actor_tokens(value):
    if not value:
        return set()
    if isinstance(value, list):
        values = value
    else:
        values = [token for token in str(value).split("@") if token]
    return {str(item).strip().lower() for item in values if str(item).strip()}


def passes_type_specific_candidate_filter(fact_text, fact_attr, old_payload, score):
    memory_type = (fact_attr or {}).get("memory_type") or old_payload.get("memory_type")
    min_score = TYPE_SPECIFIC_MIN_SCORE.get(memory_type, ADD_WITH_ATTR_SIMILARITY_THRESHOLD)
    if score < min_score:
        return False

    new_keywords = extract_fact_keywords(fact_text)
    old_keywords = extract_fact_keywords(old_payload.get("data", ""))
    keyword_overlap = len(new_keywords & old_keywords)

    new_categories = set(normalize_metadata_list((fact_attr or {}).get("category")))
    old_categories = set(normalize_metadata_list(old_payload.get("category")))
    category_overlap = bool(new_categories & old_categories)

    new_owner = normalize_actor_tokens((fact_attr or {}).get("owner"))
    old_owner = normalize_actor_tokens(old_payload.get("owner"))
    owner_overlap = bool(new_owner & old_owner)

    new_participants = normalize_actor_tokens((fact_attr or {}).get("participants"))
    old_participants = normalize_actor_tokens(old_payload.get("participants"))
    participant_overlap = bool(new_participants & old_participants)

    if memory_type == "topic":
        return keyword_overlap >= 2 or category_overlap or score >= 0.78
    if memory_type in {"statement", "decision", "knowledge"}:
        return keyword_overlap >= 2 or category_overlap or owner_overlap or score >= 0.8
    if memory_type == "todo":
        return owner_overlap or participant_overlap or keyword_overlap >= 2 or score >= 0.82
    if memory_type == "issue":
        return owner_overlap or participant_overlap or keyword_overlap >= 2 or category_overlap or score >= 0.82
    if memory_type == "summary":
        return category_overlap or keyword_overlap >= 3 or score >= 0.85
    return keyword_overlap >= 2 or score >= 0.8


def relationship_action_for_type(memory_type, relationship):
    if relationship == "CONTRADICT":
        return "contradict"
    if relationship in {"IDENTICAL", "PARAPHRASE", "LESS_COMPLETE"}:
        return "weight"
    if relationship == "MORE_COMPLETE":
        if memory_type in {"statement", "decision", "knowledge"}:
            return "embedding"
        return "weight"
    return "weight"


def select_primary_relationships(relationships, all_old_memories):
    if not relationships:
        return []

    def _relationship_sort_key(rel):
        old_fact_id = rel.get("old_fact_id")
        relationship = rel.get("relationship", "")
        candidate_score = 0.0
        if old_fact_id in all_old_memories:
            candidate_score = all_old_memories[old_fact_id].get("score", 0.0)
        return (
            ADD_WITH_ATTR_RELATIONSHIP_PRIORITY.get(relationship, 0),
            candidate_score,
        )

    selected = max(relationships, key=_relationship_sort_key)
    return [selected]
