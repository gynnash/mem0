"""Source-local extraction through an injected structured ModelPort."""

import json
import re
from dataclasses import dataclass
from typing import Dict

from mem0.v3.extraction.models import (
    EvidenceSpan,
    EvidenceUnit,
    ExtractedClaim,
    ExtractedProjectMention,
    LocalExtractionResult,
    MeetingExtractionInput,
    SessionTopicCandidate,
    TranscriptSegment,
    UnitBackedLocalExtractionResult,
)
from mem0.v3.extraction.prompts import LOCAL_EXTRACTION_SYSTEM_PROMPT
from mem0.v3.ports import ModelMessage, ModelPort, StructuredModelRequest


class ExtractionValidationError(ValueError):
    pass


MAX_EVIDENCE_UNIT_CHARS = 240
EVIDENCE_UNIT_BOUNDARY_RE = re.compile(r"[。！？!?；;，,\n]+")


@dataclass(frozen=True)
class _MaterializedEvidenceUnit:
    unit: EvidenceUnit
    start_char: int
    end_char: int


class LocalExtractionService:
    def __init__(self, model: ModelPort, *, timeout_ms: int = 60_000) -> None:
        self._model = model
        self._timeout_ms = timeout_ms

    def extract(self, source: MeetingExtractionInput) -> LocalExtractionResult:
        evidence_units = self._split_transcript(source.segments)
        transcript = [
            {
                "segment_id": segment.segment_id,
                "speaker_ref": segment.speaker_ref,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "evidence_units": [
                    {
                        "evidence_unit_id": item.unit.evidence_unit_id,
                        "text": item.unit.text,
                    }
                    for item in evidence_units.values()
                    if item.unit.segment_id == segment.segment_id
                ],
            }
            for segment in source.segments
        ]
        unit_backed_response = self._model.generate_structured(
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
            response_model=UnitBackedLocalExtractionResult,
        )
        response = self._materialize_evidence_spans(
            unit_backed_response, evidence_units
        )
        self._validate_spans(source.segments, response)
        return response

    @classmethod
    def _split_transcript(
        cls, segments: tuple[TranscriptSegment, ...]
    ) -> Dict[str, _MaterializedEvidenceUnit]:
        units = {}
        for segment in segments:
            for item in cls._split_segment(segment):
                units[item.unit.evidence_unit_id] = item
        return units

    @staticmethod
    def _split_segment(
        segment: TranscriptSegment,
    ) -> tuple[_MaterializedEvidenceUnit, ...]:
        ranges = []
        start = 0
        for match in EVIDENCE_UNIT_BOUNDARY_RE.finditer(segment.text):
            LocalExtractionService._append_bounded_ranges(
                ranges, segment.text, start, match.end()
            )
            start = match.end()
        LocalExtractionService._append_bounded_ranges(
            ranges, segment.text, start, len(segment.text)
        )
        if not ranges:
            ranges.append((0, len(segment.text)))
        return tuple(
            _MaterializedEvidenceUnit(
                unit=EvidenceUnit(
                    evidence_unit_id=f"{segment.segment_id}:u{index}",
                    segment_id=segment.segment_id,
                    text=segment.text[start_char:end_char],
                ),
                start_char=start_char,
                end_char=end_char,
            )
            for index, (start_char, end_char) in enumerate(ranges)
        )

    @staticmethod
    def _append_bounded_ranges(ranges, text: str, start: int, end: int) -> None:
        while start < end:
            bounded_end = min(start + MAX_EVIDENCE_UNIT_CHARS, end)
            content_start = start
            content_end = bounded_end
            while content_start < content_end and text[content_start].isspace():
                content_start += 1
            while content_end > content_start and text[content_end - 1].isspace():
                content_end -= 1
            if content_start < content_end:
                ranges.append((content_start, content_end))
            start = bounded_end

    @classmethod
    def _materialize_evidence_spans(
        cls,
        result: UnitBackedLocalExtractionResult,
        evidence_units: Dict[str, _MaterializedEvidenceUnit],
    ) -> LocalExtractionResult:
        return LocalExtractionResult(
            extraction_version=result.extraction_version,
            claims=tuple(
                ExtractedClaim(
                    **item.model_dump(exclude={"evidence_unit_ids"}),
                    evidence_spans=cls._spans_for_unit_ids(
                        item.evidence_unit_ids, evidence_units
                    ),
                )
                for item in result.claims
            ),
            project_mentions=tuple(
                ExtractedProjectMention(
                    **item.model_dump(exclude={"evidence_unit_ids"}),
                    evidence_spans=cls._spans_for_unit_ids(
                        item.evidence_unit_ids, evidence_units
                    ),
                )
                for item in result.project_mentions
            ),
            topic_candidates=tuple(
                SessionTopicCandidate(
                    **item.model_dump(exclude={"evidence_unit_ids"}),
                    evidence_spans=cls._spans_for_unit_ids(
                        item.evidence_unit_ids, evidence_units
                    ),
                )
                for item in result.topic_candidates
            ),
            warnings=result.warnings,
        )

    @staticmethod
    def _spans_for_unit_ids(
        unit_ids: tuple[str, ...],
        evidence_units: Dict[str, _MaterializedEvidenceUnit],
    ) -> tuple[EvidenceSpan, ...]:
        spans = []
        for unit_id in dict.fromkeys(unit_ids):
            materialized = evidence_units.get(unit_id)
            if materialized is None:
                raise ExtractionValidationError(
                    f"extraction references unknown evidence unit: {unit_id}"
                )
            spans.append(
                EvidenceSpan(
                    segment_id=materialized.unit.segment_id,
                    start_char=materialized.start_char,
                    end_char=materialized.end_char,
                )
            )
        return tuple(spans)

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
