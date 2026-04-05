from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class ExtractedPage:
    """Represents one logical page (or the whole document if pages are not applicable)."""

    page_number: int  # 1-based; 0 means "no page concept"
    text: str
    extra_metadata: dict = field(default_factory=dict)


class BaseExtractor(ABC):
    """All file-type extractors implement this interface."""

    @abstractmethod
    def extract(self, file_path: Path) -> Iterator[ExtractedPage]:
        """Yield ExtractedPage objects, one per logical page / section."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return lowercase extensions this extractor handles, e.g. ['.pdf']."""
        ...
