"""Document classification rules."""

from __future__ import annotations

from .models import ClassificationResult, DocumentType


class DocumentClassifier:
    """Rule-based document classifier."""

    KEYWORDS: dict[DocumentType, tuple[str, ...]] = {
        DocumentType.INVOICE: ("invoice", "rechnung", "faktura", "payment"),
        DocumentType.UTILITIES: ("utilities", "nebenkosten", "betriebskosten", "heating"),
        DocumentType.PAYMENT_REMINDER: ("mahnung", "reminder", "payment reminder"),
        DocumentType.CONTRACT: ("vertrag", "contract", "vereinbarung"),
        DocumentType.FORM: ("formular", "form", "antrag", "application"),
        DocumentType.LETTER: ("brief", "letter", "mitteilung"),
    }

    def classify(self, filename: str, text: str) -> ClassificationResult:
        """Classify a document using filename and text keyword matches."""
        filename_lower = filename.lower()
        text_lower = text.lower()

        scores: dict[DocumentType, float] = {}
        for doc_type, terms in self.KEYWORDS.items():
            score = 0.0
            for term in terms:
                if term in filename_lower:
                    score += 2.0
                if term in text_lower:
                    score += 1.0
            scores[doc_type] = score

        best_type = max(scores, key=lambda doc_type: scores[doc_type])
        max_score = scores[best_type]
        if max_score <= 0:
            return ClassificationResult(document_type=DocumentType.OTHER, confidence=0.0)

        return ClassificationResult(
            document_type=best_type,
            confidence=min(1.0, max_score / 5.0),
        )
