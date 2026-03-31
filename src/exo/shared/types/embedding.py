"""Canonical internal type for text embedding task parameters."""

from typing import Literal

from pydantic import BaseModel

from exo.shared.types.common import ModelId


class TextEmbeddingTaskParams(BaseModel, frozen=True):
    """Internal task params for embedding inference.

    The API endpoint converts the wire type into this before handing
    off to the master/worker pipeline.
    """

    model: ModelId
    input_texts: list[str]
    encoding_format: Literal["float", "base64"] = "float"
