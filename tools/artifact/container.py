"""Minimal read-only support for the GInfer v2 object directory."""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

from .layouts import align_up, encoded_size, get_layout


MAGIC = b"NINFER\x00\x02"
_V1_MAGIC = b"NINFER\x00\x01"
PREFIX = struct.Struct("<8sQ")
PREFIX_BYTES = PREFIX.size
PAYLOAD_ALIGNMENT = 4096
RAW_BYTES_V1 = "raw-bytes-v1"

_ROOT_MEMBERS = frozenset({"identity", "objects"})
_IDENTITY_MEMBERS = frozenset({"model_id", "weights_id"})
_TENSOR_MEMBERS = frozenset(
    {"name", "kind", "shape", "format", "layout", "offset", "bytes"}
)
_RESOURCE_MEMBERS = frozenset({"name", "kind", "encoding", "offset", "bytes"})


class ArtifactError(ValueError):
    """The file does not satisfy the GInfer v2 directory contract."""


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    model_id: str
    weights_id: str


@dataclass(frozen=True, slots=True)
class TensorObject:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str
    offset: int
    bytes: int

    @property
    def kind(self) -> str:
        return "tensor"


@dataclass(frozen=True, slots=True)
class ResourceObject:
    name: str
    encoding: str
    offset: int
    bytes: int

    @property
    def kind(self) -> str:
        return "resource"


ArtifactObject: TypeAlias = TensorObject | ResourceObject


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{field} must be a nonempty string")
    return value


def _require_integer(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ArtifactError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ArtifactError(f"{field} must be {qualifier}")
    return value


def _require_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ArtifactError("tensor shape must be an array")
    return tuple(_require_integer(dim, "shape dimension", positive=True) for dim in value)


def _resource_alignment(encoding: str) -> int:
    if encoding != RAW_BYTES_V1:
        raise ArtifactError(f"unknown resource encoding: {encoding}")
    return 1


def object_alignment(obj: ArtifactObject) -> int:
    if isinstance(obj, TensorObject):
        return get_layout(obj.layout).alignment
    return _resource_alignment(obj.encoding)


def _parse_object(value: object) -> ArtifactObject:
    if not isinstance(value, dict):
        raise ArtifactError("each object entry must be a JSON object")
    kind = value.get("kind")
    if kind == "tensor":
        if frozenset(value) != _TENSOR_MEMBERS:
            raise ArtifactError("tensor entry has missing or extra members")
        name = _require_string(value["name"], "tensor name")
        shape = _require_shape(value["shape"])
        format_name = _require_string(value["format"], "tensor format")
        layout_name = _require_string(value["layout"], "tensor layout")
        offset = _require_integer(value["offset"], "tensor offset")
        payload_bytes = _require_integer(value["bytes"], "tensor bytes", positive=True)
        try:
            expected = encoded_size(layout_name, format_name, shape)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(str(exc)) from exc
        if payload_bytes != expected:
            raise ArtifactError(
                f"tensor {name} stores {payload_bytes} bytes; layout requires {expected}"
            )
        return TensorObject(name, shape, format_name, layout_name, offset, payload_bytes)
    if kind == "resource":
        if frozenset(value) != _RESOURCE_MEMBERS:
            raise ArtifactError("resource entry has missing or extra members")
        name = _require_string(value["name"], "resource name")
        encoding = _require_string(value["encoding"], "resource encoding")
        _resource_alignment(encoding)
        offset = _require_integer(value["offset"], "resource offset")
        payload_bytes = _require_integer(value["bytes"], "resource bytes", positive=True)
        return ResourceObject(name, encoding, offset, payload_bytes)
    raise ArtifactError("object kind must be 'tensor' or 'resource'")


def parse_directory(
    data: bytes,
) -> tuple[ArtifactIdentity, tuple[ArtifactObject, ...]]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON directory: {exc}") from exc
    if not isinstance(value, dict) or frozenset(value) != _ROOT_MEMBERS:
        raise ArtifactError("directory root must contain exactly identity and objects")
    raw_identity = value["identity"]
    if (
        not isinstance(raw_identity, dict)
        or frozenset(raw_identity) != _IDENTITY_MEMBERS
    ):
        raise ArtifactError(
            "artifact identity must contain exactly model_id and weights_id"
        )
    identity = ArtifactIdentity(
        model_id=_require_string(raw_identity["model_id"], "model_id"),
        weights_id=_require_string(raw_identity["weights_id"], "weights_id"),
    )
    raw_objects = value["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ArtifactError("objects must be a nonempty array")
    objects = tuple(_parse_object(item) for item in raw_objects)
    names: set[str] = set()
    for obj in objects:
        if obj.name in names:
            raise ArtifactError(f"duplicate object name: {obj.name}")
        names.add(obj.name)
    return identity, objects


def _validate_ranges(
    objects: Sequence[ArtifactObject], payload_bytes: int
) -> dict[str, ArtifactObject]:
    cursor = 0
    index: dict[str, ArtifactObject] = {}
    for obj in objects:
        alignment = object_alignment(obj)
        if obj.offset < cursor:
            raise ArtifactError(f"object {obj.name} overlaps or is out of order")
        if obj.offset % alignment:
            raise ArtifactError(f"object {obj.name} is not {alignment}-byte aligned")
        end = obj.offset + obj.bytes
        if end > payload_bytes:
            raise ArtifactError(f"object {obj.name} extends beyond the file")
        cursor = end
        index[obj.name] = obj
    return index


class Artifact:
    """Mmap-backed, structurally validated `.ginfer` artifact."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        try:
            self._file.seek(0, 2)
            self.file_bytes = self._file.tell()
            self._file.seek(0)
            if self.file_bytes < PREFIX_BYTES:
                raise ArtifactError("artifact is shorter than the v2 prefix")
            prefix = self._file.read(PREFIX_BYTES)
            magic, json_bytes = PREFIX.unpack(prefix)
            if magic == _V1_MAGIC:
                raise ArtifactError(
                    "GInfer artifact v1 is no longer supported; replace it with the "
                    "published version-2 artifact"
                )
            if magic != MAGIC:
                raise ArtifactError("artifact magic is not GInfer v2")
            if json_bytes == 0:
                raise ArtifactError("json_bytes must be positive")
            metadata_end = PREFIX_BYTES + json_bytes
            self.payload_offset = align_up(metadata_end, PAYLOAD_ALIGNMENT)
            if metadata_end > self.file_bytes or self.payload_offset > self.file_bytes:
                raise ArtifactError("declared JSON or payload start extends beyond the file")
            directory = self._file.read(json_bytes)
            if len(directory) != json_bytes:
                raise ArtifactError("artifact JSON is truncated")
            self.identity, self.objects = parse_directory(directory)
            payload_bytes = self.file_bytes - self.payload_offset
            self._index = _validate_ranges(self.objects, payload_bytes)
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            if self._mapping is not None:
                self._mapping.close()
            self._file.close()
            raise

    @classmethod
    def open(cls, path: str | Path) -> "Artifact":
        return cls(path)

    def find(self, name: str) -> ArtifactObject:
        return self._index[name]

    def payload(self, obj: ArtifactObject | str) -> memoryview:
        if isinstance(obj, str):
            obj = self.find(obj)
        if self._mapping is None:
            raise RuntimeError("artifact is closed")
        begin = self.payload_offset + obj.offset
        return memoryview(self._mapping)[begin : begin + obj.bytes]

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "Artifact":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
