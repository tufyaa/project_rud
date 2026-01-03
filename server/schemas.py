from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class VectorizeRequest(BaseModel):
    texts: list[str] = Field(..., description="Input documents")
    lower: bool = Field(True, description="Lowercase tokens")
    min_token_len: int = Field(2, description="Minimum token length")


class TfidfRequest(VectorizeRequest):
    smooth_idf: bool = Field(True, description="Apply smooth IDF")
    normalize: Literal["l2", "none"] = Field("l2", description="Normalization type")


class LsaRequest(VectorizeRequest):
    n_components: int = Field(2, description="Number of SVD components")
    n_top_terms: int = Field(10, description="Number of top terms per component")


class TextRequest(BaseModel):
    text: str = Field(..., description="Input text")
