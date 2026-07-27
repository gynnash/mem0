"""Source-local extraction through an injected structured ModelPort."""

import json
from typing import Dict

from mem0.v3.extraction.models import (
    EvidenceSpan,
    LocalExtractionResult,
    MeetingExtractionInput,
    TranscriptSegment,
)
from mem0.v3.extraction.prompts import LOCAL_EXTRACTION_SYSTEM_PROMPT
from mem0.v3.ports import ModelMessage, ModelPort, StructuredModelRequest


class ExtractionValidationError(ValueError):
    pass


class LocalExtractionService:
    def __init__(self, model: ModelPort, *, timeout_ms: int = 60_000) -> None:
        self._model = model
        self._timeout_ms = timeout_ms

    def extract(self, source: MeetingExtractionInput) -> LocalExtractionResult:
        transcript = [
            {
                "segment_id": segment.segment_id,
                "speaker_ref": segment.speaker_ref,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.text,
            }
            for segment in source.segments
        ]
        response = self._model.generate_structured(
            request=StructuredModelRequest(
                operation="memory_v3.local_extraction",
                timeout_ms=self._timeout_ms,
                messages=(
                    ModelMessage(role="system", content=LOCAL_EXTRACTION_SYSTEM_PROMPT),
                    ModelMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "meeting": {
                                    "memory_id": source.memory_id,
                                    "title": source.title,
                                    "started_at": source.started_at.isoformat(),
                                    "ended_at": (
                                        source.ended_at.isoformat()
                                        if source.ended_at is not None
                                        else None
                                    ),
                                    "participant_refs": source.participant_refs,
                                },
                                "transcript_segments": transcript,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ),
                metadata={
                    "memory_id": source.memory_id,
                    "transcript_version": source.transcript_version,
                },
            ),
            response_model=LocalExtractionResult,
        )
        self._validate_spans(source.segments, response)
        return response

    @staticmethod
    def _validate_spans(
        segments: tuple[TranscriptSegment, ...], result: LocalExtractionResult
    ) -> None:
        by_id: Dict[str, TranscriptSegment] = {item.segment_id: item for item in segments}
        spans = []
        spans.extend(
            span for claim in result.claims for span in claim.evidence_spans
        )
        spans.extend(
            span
            for mention in result.project_mentions
            for span in mention.evidence_spans
        )
        spans.extend(
            span
            for topic in result.topic_candidates
            for span in topic.evidence_spans
        )
        for span in spans:
            LocalExtractionService._validate_span(by_id, span)

    @staticmethod
    def _validate_span(
        segments: Dict[str, TranscriptSegment], span: EvidenceSpan
    ) -> None:
        segment = segments.get(span.segment_id)
        if segment is None:
            raise ExtractionValidationError(
                f"evidence references unknown segment: {span.segment_id}"
            )
        if span.end_char > len(segment.text):
            raise ExtractionValidationError(
                f"evidence span exceeds segment length: {span.segment_id}"
            )

