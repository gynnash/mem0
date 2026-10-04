"""Role fields shared by action planning and identity correction."""

ACTION_ROLES = {
    "executor": ("owner_mention", "owner_speaker_ref", "owner_entity_id", "owner_evidence_ids"),
    "initiator": ("initiator_mention", "initiator_speaker_ref", "initiator_entity_id", "initiator_evidence_ids"),
}
EXECUTOR_FIELDS = {*ACTION_ROLES["executor"], "committed_by"}
ROLE_ENTITY_FIELDS = {fields[2] for fields in ACTION_ROLES.values()}
ROLE_SPEAKER_FIELDS = {fields[1] for fields in ACTION_ROLES.values()} | {"asserted_by_speaker_ref"}
ROLE_MENTION_FIELDS = {fields[0] for fields in ACTION_ROLES.values()}
ROLE_EVIDENCE_FIELDS = {fields[3] for fields in ACTION_ROLES.values()}
