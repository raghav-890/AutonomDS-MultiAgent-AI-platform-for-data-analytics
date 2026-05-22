"""
AutonomDS File & Upload Validator
===================================
Security-first validation for all uploaded datasets.
Enforces type, size, content, and schema constraints.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("validators")
settings = get_settings()

# ── MIME type → allowed extensions mapping ────────────────────────────────────
ALLOWED_MIME_TYPES: dict[str, list[str]] = {
    "text/csv": ["csv"],
    "application/csv": ["csv"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ["xlsx"],
    "application/vnd.ms-excel": ["xls"],
    "application/octet-stream": ["parquet", "sqlite", "db", "csv", "xlsx"],
    "application/x-sqlite3": ["sqlite", "db"],
}

# Magic bytes for binary format detection
MAGIC_BYTES: dict[str, bytes] = {
    "xlsx": b"PK\x03\x04",            # ZIP header (XLSX is ZIP)
    "parquet": b"PAR1",                # Parquet magic
    "sqlite": b"SQLite format 3\x00",  # SQLite magic
}


class ValidationError(Exception):
    """Raised when a file fails validation."""

    def __init__(self, message: str, field: str = "file") -> None:
        super().__init__(message)
        self.field = field
        self.message = message


class FileValidator:
    """Validates uploaded dataset files for security and correctness."""

    def __init__(
        self,
        max_size_bytes: Optional[int] = None,
        allowed_extensions: Optional[list[str]] = None,
    ) -> None:
        self.max_size_bytes = max_size_bytes or settings.max_upload_size_bytes
        self.allowed_extensions = allowed_extensions or settings.allowed_extensions

    def validate_extension(self, filename: str) -> str:
        """Validate file extension. Returns normalized extension."""
        ext = Path(filename).suffix.lstrip(".").lower()
        if not ext:
            raise ValidationError("File has no extension.", "filename")
        if ext not in self.allowed_extensions:
            raise ValidationError(
                f"Extension '.{ext}' not allowed. "
                f"Allowed: {', '.join(self.allowed_extensions)}",
                "extension",
            )
        return ext

    def validate_size(self, content: bytes) -> None:
        """Validate file size does not exceed limit."""
        size = len(content)
        if size == 0:
            raise ValidationError("File is empty.", "size")
        if size > self.max_size_bytes:
            mb = size / (1024 * 1024)
            max_mb = self.max_size_bytes / (1024 * 1024)
            raise ValidationError(
                f"File size ({mb:.1f} MB) exceeds limit ({max_mb:.0f} MB).", "size"
            )

    def validate_magic_bytes(self, content: bytes, ext: str) -> None:
        """Validate binary file magic bytes match declared extension."""
        if ext in MAGIC_BYTES:
            magic = MAGIC_BYTES[ext]
            if not content.startswith(magic):
                raise ValidationError(
                    f"File content does not match declared type '.{ext}'. "
                    "Possible file spoofing attempt.",
                    "content",
                )

    def validate_csv_content(self, content: bytes) -> pd.DataFrame:
        """Parse and validate CSV content. Returns DataFrame."""
        try:
            # Try common encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValidationError("Cannot decode CSV file encoding.", "encoding")
        except pd.errors.EmptyDataError:
            raise ValidationError("CSV file is empty or has no data.", "content")
        except pd.errors.ParserError as e:
            raise ValidationError(f"CSV parsing failed: {e}", "content")

        self._validate_dataframe(df)
        return df

    def validate_excel_content(self, content: bytes) -> pd.DataFrame:
        """Parse and validate Excel content. Returns DataFrame."""
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=0)
        except Exception as e:
            raise ValidationError(f"Excel parsing failed: {e}", "content")
        self._validate_dataframe(df)
        return df

    def validate_parquet_content(self, content: bytes) -> pd.DataFrame:
        """Parse and validate Parquet content. Returns DataFrame."""
        try:
            df = pd.read_parquet(io.BytesIO(content))
        except Exception as e:
            raise ValidationError(f"Parquet parsing failed: {e}", "content")
        self._validate_dataframe(df)
        return df

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Common DataFrame validations."""
        if df.empty:
            raise ValidationError("Dataset has no rows.", "content")
        if len(df.columns) < 2:
            raise ValidationError(
                "Dataset must have at least 2 columns.", "content"
            )
        if len(df) < 5:
            raise ValidationError(
                "Dataset must have at least 5 rows for analysis.", "content"
            )
        # Check for suspicious column names (SQL injection etc.)
        for col in df.columns:
            col_str = str(col)
            if any(c in col_str for c in [";", "--", "/*", "*/"]):
                raise ValidationError(
                    f"Suspicious column name detected: '{col_str}'", "content"
                )

    def validate(
        self, filename: str, content: bytes
    ) -> tuple[str, pd.DataFrame | None]:
        """
        Run full validation pipeline.

        Returns:
            Tuple of (extension, DataFrame | None for non-CSV/Excel/Parquet)

        Raises:
            ValidationError: On any validation failure
        """
        logger.info("validating_file", filename=filename, size_bytes=len(content))

        ext = self.validate_extension(filename)
        self.validate_size(content)
        self.validate_magic_bytes(content, ext)

        df: pd.DataFrame | None = None
        if ext == "csv":
            df = self.validate_csv_content(content)
        elif ext in ("xlsx", "xls"):
            df = self.validate_excel_content(content)
        elif ext == "parquet":
            df = self.validate_parquet_content(content)
        elif ext in ("sqlite", "db"):
            # SQLite validated by magic bytes above; full parse in ingestion agent
            pass

        logger.info(
            "file_validated",
            filename=filename,
            ext=ext,
            rows=len(df) if df is not None else "N/A",
            cols=len(df.columns) if df is not None else "N/A",
        )
        return ext, df

    @staticmethod
    def compute_checksum(content: bytes) -> str:
        """SHA-256 checksum for deduplication and integrity checks."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def safe_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks."""
        # Keep only safe characters
        safe = "".join(
            c for c in Path(filename).name if c.isalnum() or c in "._- "
        )
        return safe[:255] or "uploaded_file"


# ── Module-level convenience ──────────────────────────────────────────────────
_default_validator = FileValidator()


def validate_upload(filename: str, content: bytes) -> tuple[str, pd.DataFrame | None]:
    """Convenience function using default validator settings."""
    return _default_validator.validate(filename, content)
