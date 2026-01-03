"""Pydantic schemas for API requests."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class VectorizeRequest(BaseModel):
    """Common payload for vectorization endpoints."""
    texts: list[str] = Field(..., description="Input documents")
    lower: bool = Field(True, description="Lowercase tokens")
    min_token_len: int = Field(2, description="Minimum token length")


class TfidfRequest(VectorizeRequest):
    """Payload for TF-IDF endpoint."""
    smooth_idf: bool = Field(True, description="Apply smooth IDF")
    normalize: Literal["l2", "none"] = Field("l2", description="Normalization type")


class LsaRequest(VectorizeRequest):
    """Payload for LSA endpoint."""
    n_components: int = Field(2, description="Number of SVD components")
    n_top_terms: int = Field(10, description="Number of top terms per component")


class TextRequest(BaseModel):
    """Payload for NLTK text operations."""
    text: str = Field(..., description="Input text")
