from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class OAISchemaBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
