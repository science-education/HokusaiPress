class PDFExportError(Exception):
    """Base exception for PDF/MRC export operations."""
    pass


class UnsupportedCodecError(PDFExportError):
    """Raised when an unsupported image codec or mode is requested."""
    pass


class MissingFontError(PDFExportError):
    """Raised when the required font cannot be discovered or loaded."""
    pass


class BackendUnavailableError(PDFExportError):
    """Raised when the requested PDF export backend is not available."""
    pass
