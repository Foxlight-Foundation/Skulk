"""Journaled stopped-owner runtime selection with retained logical plugin state."""

import os
from pathlib import Path
from typing import Literal, Self, final
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

from skulk.extensions.runtime_artifacts import (
    Digest,
    Identifier,
    QualifiedHost,
    RuntimePlatform,
    VerifiedRuntime,
    canonical_json,
)
from skulk.extensions.runtime_files import (
    RuntimeLock,
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_install import RuntimeInstaller

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class RuntimeSelection(_Contract):
    """Atomic desired runtime selection, independent of observed process health."""

    protocol: Literal[1] = 1
    revision: int = Field(
        ge=1, description="Monotonic installation selection revision."
    )
    operation_id: str = Field(
        pattern=r"^[a-f0-9]{32}$", description="Operation that selected this runtime."
    )
    runtime_digest: Digest = Field(description="Exact retained signed generation.")
    enabled: bool = Field(
        description="Whether the independent owner may serve this installation."
    )
    bundle_id: Identifier = Field(
        description="Stable plugin identity across generations."
    )
    sequence: int = Field(ge=1, description="Selected publisher release sequence.")
    highest_sequence: int = Field(
        ge=1,
        description="Highest previously selected sequence, retained across rollback.",
    )
    state_schema: Identifier = Field(description="Current durable plugin state schema.")
    permissions: tuple[str, ...] = Field(
        max_length=16, description="Permissions explicitly accepted for this selection."
    )
    configuration_schema: JsonValue = Field(
        description="Opaque signed configuration schema used for migration compatibility."
    )
    platform: RuntimePlatform = Field(
        description="Locally measured qualified platform."
    )
    python_version: str = Field(
        max_length=128, description="Locally measured base Python version."
    )
    skulk_version: str = Field(
        max_length=128, description="Locally measured qualified Skulk version."
    )
    skulk_build_sha256: Digest = Field(
        description="Locally measured qualified Skulk build."
    )


class SelectionOperation(_Contract):
    """Durable local selection intent and safe reconnect/recovery status."""

    operation_id: str = Field(
        pattern=r"^[a-f0-9]{32}$", description="Immutable local operation ID."
    )
    state: Literal["pending", "complete", "recovery_required"] = Field(
        description="Selection progress, never a provider create operation."
    )
    expected_revision: int = Field(
        ge=0, description="Owner-reviewed previous selection revision."
    )
    selection: RuntimeSelection = Field(
        description="Exact authorized target; no paths or credentials."
    )

    @model_validator(mode="after")
    def bounded(self) -> Self:
        """Reject intent that could not be read back within the journal bound."""
        if len(canonical_json(self.model_dump(mode="json"))) > 131072:
            raise ValueError("selection intent exceeds bound")
        if self.operation_id != self.selection.operation_id:
            raise ValueError("selection operation identity differs")
        return self


def _remove_private(path: Path) -> None:
    path.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest(runtime: VerifiedRuntime) -> dict[str, JsonValue]:
    payload = _JSON_OBJECT.validate_json(runtime.payload)
    release = _JSON_OBJECT.validate_python(payload["release"], strict=True)
    return _JSON_OBJECT.validate_python(release["manifest"], strict=True)


@final
class RuntimeSelector:
    """Select or withdraw one isolated runtime only while its owner is stopped.

    The final selection is one fsynced atomic file. Intent and operation results
    are retained separately so reconnect cannot repeat a change. Recovery only
    completes this authorized local file transition; it never invokes a provider.
    Logical identities, configuration, credentials and cleanup remain untouched.
    """

    def __init__(self, root: Path) -> None:
        """Use protected installation state shared with the generic installer."""
        self.installer = RuntimeInstaller(root)
        self.root = self.installer.root
        self.records = self.installer.installer / "selections"
        private_directory(self.records)
        self.pending = self.records / "pending.json"

    def current(self) -> RuntimeSelection | None:
        """Read the selected runtime without inferring that its process is healthy."""
        try:
            raw = read_private(self.root / "runtime-selection.json")
        except FileNotFoundError:
            return None
        return RuntimeSelection.model_validate_json(raw)

    def operation(self, operation_id: str) -> SelectionOperation:
        """Read retained progress; never launch or replay a local operation."""
        identifier = TypeAdapter(str).validate_python(operation_id, strict=True)
        if len(identifier) != 32 or any(
            c not in "0123456789abcdef" for c in identifier
        ):
            raise ValueError("invalid selection operation ID")
        try:
            raw = read_private(self.records / (identifier + ".json"))
        except FileNotFoundError:
            raw = read_private(self.pending)
        operation = SelectionOperation.model_validate_json(raw)
        if operation.operation_id != identifier:
            raise ValueError("selection operation identity differs")
        return operation

    def _save(self, operation: SelectionOperation) -> None:
        write_private(
            self.records / (operation.operation_id + ".json"),
            canonical_json(operation.model_dump(mode="json")),
        )

    def _apply(self, operation: SelectionOperation) -> SelectionOperation:
        current = self.current()
        if current != operation.selection and (
            (current.revision if current else 0) != operation.expected_revision
        ):
            raise ValueError("selection revision conflict")
        # A legacy installation needs the explicit stopped-owner migration, not
        # a second source of manifests that could run an unintended sibling.
        if any(self.root.glob("*.manifest.json")):
            raise ValueError("legacy manifests require explicit migration")
        write_private(self.pending, canonical_json(operation.model_dump(mode="json")))
        try:
            self._save(operation)
            write_private(
                self.root / "runtime-selection.json",
                canonical_json(operation.selection.model_dump(mode="json")),
            )
            complete = operation.model_copy(update={"state": "complete"})
            self._save(complete)
            _remove_private(self.pending)
            return complete
        except OSError:
            self._save(operation.model_copy(update={"state": "recovery_required"}))
            raise

    def _start(
        self, selection: RuntimeSelection, expected_revision: int
    ) -> SelectionOperation:
        if self.pending.exists():
            raise ValueError("pending selection requires recovery")
        try:
            prior = self.operation(selection.operation_id)
        except FileNotFoundError:
            prior = None
        if prior is not None:
            if (
                prior.selection != selection
                or prior.expected_revision != expected_revision
            ):
                raise ValueError("selection operation identity conflict")
            if prior.state != "complete":
                raise ValueError("selection operation requires recovery")
            return prior
        return self._apply(
            SelectionOperation(
                operation_id=selection.operation_id,
                state="pending",
                expected_revision=expected_revision,
                selection=selection,
            )
        )

    def _candidate(
        self,
        runtime: VerifiedRuntime,
        host: QualifiedHost,
        *,
        expected_revision: int,
        operation_id: str,
        rollback: bool,
        accept_permissions: bool,
    ) -> RuntimeSelection:
        current = self.current()
        release = runtime.claims.release
        schema = _manifest(runtime).get("configuration_schema")
        if current is not None:
            if current.bundle_id != release.manifest.bundle_id:
                raise ValueError("installation bundle identity differs")
            if release.sequence < current.highest_sequence and not rollback:
                raise ValueError("rollback requires explicit selection")
            if (
                set(release.permissions) - set(current.permissions)
                and not accept_permissions
            ):
                raise ValueError("expanded permissions require acceptance")
            if (
                current.state_schema != release.state_schema
                and current.state_schema not in release.compatible_state_schemas
            ):
                raise ValueError("incompatible state migration")
            if current.configuration_schema != schema:
                raise ValueError("configuration schema requires migration")
        elif rollback:
            raise ValueError("rollback requires an existing selection")
        return RuntimeSelection(
            revision=expected_revision + 1,
            operation_id=operation_id or uuid4().hex,
            runtime_digest=runtime.digest,
            enabled=True,
            bundle_id=release.manifest.bundle_id,
            sequence=release.sequence,
            highest_sequence=max(
                current.highest_sequence if current else 0, release.sequence
            ),
            state_schema=release.state_schema,
            permissions=release.permissions,
            configuration_schema=schema,
            platform=host.platform,
            python_version=host.python_version,
            skulk_version=host.skulk_version,
            skulk_build_sha256=host.skulk_build_sha256,
        )

    async def preview(
        self,
        runtime_digest: str,
        *,
        expected_revision: int,
        operation_id: str,
        rollback: bool = False,
        accept_permissions: bool = False,
    ) -> RuntimeSelection:
        """Validate an exact local selection without interrupting a running owner.

        This nonbillable inspection holds installer ownership, verifies the target
        and checks revision, permissions and migration compatibility. Activation
        repeats these checks under stopped-owner ownership before publication.
        """
        async with self.installer.locked_generation(runtime_digest) as (runtime, host):
            current = self.current()
            if (current.revision if current else 0) != expected_revision:
                raise ValueError("selection revision conflict")
            return self._candidate(
                runtime,
                host,
                expected_revision=expected_revision,
                operation_id=operation_id,
                rollback=rollback,
                accept_permissions=accept_permissions,
            )

    async def activate(
        self,
        runtime_digest: str,
        *,
        expected_revision: int,
        operation_id: str | None = None,
        rollback: bool = False,
        accept_permissions: bool = False,
    ) -> SelectionOperation:
        """Select a verified generation after the caller stops its private owner.

        Expanded permissions and lower release sequences require explicit owner
        choices. State compatibility is publisher-declared; configuration schema
        changes require migration tooling. This does not approve paid proposals.
        """
        async with self.installer.locked_generation(runtime_digest) as (runtime, host):
            owner = RuntimeLock(self.root, "supervisor.lock")
            try:
                selection = self._candidate(
                    runtime,
                    host,
                    expected_revision=expected_revision,
                    operation_id=operation_id or uuid4().hex,
                    rollback=rollback,
                    accept_permissions=accept_permissions,
                )
                return self._start(selection, expected_revision)
            finally:
                owner.close()

    def disable(
        self, *, expected_revision: int, operation_id: str | None = None
    ) -> SelectionOperation:
        """Withdraw a stopped installation even if release trust is now invalid.

        Retain its selected version, runtime, logical state and cleanup material.
        An uninstall uses this same retained-state withdrawal in v1.
        """
        installer = RuntimeLock(self.installer.installer)
        try:
            owner = RuntimeLock(self.root, "supervisor.lock")
            try:
                current = self.current()
                if current is None:
                    raise ValueError("no selected runtime")
                selection = current.model_copy(
                    update={
                        "revision": expected_revision + 1,
                        "operation_id": operation_id or uuid4().hex,
                        "enabled": False,
                    }
                )
                # model_copy is only for internal immutable updates; validate
                # the externally supplied revision and operation ID at the edge.
                selection = RuntimeSelection.model_validate_json(
                    selection.model_dump_json()
                )
                return self._start(selection, expected_revision)
            finally:
                owner.close()
        finally:
            installer.close()

    async def recover(self) -> SelectionOperation:
        """Finish one journaled local selection after interruption and revalidation."""
        operation = SelectionOperation.model_validate_json(read_private(self.pending))
        if operation.selection.enabled:
            async with self.installer.locked_generation(
                operation.selection.runtime_digest
            ) as (_, host):
                expected = operation.selection
                if host != QualifiedHost(
                    expected.platform,
                    expected.python_version,
                    expected.skulk_version,
                    expected.skulk_build_sha256,
                ):
                    raise ValueError("recovery host differs")
                return self._recover_locked(operation)
        installer = RuntimeLock(self.installer.installer)
        try:
            return self._recover_locked(operation)
        finally:
            installer.close()

    def _recover_locked(self, operation: SelectionOperation) -> SelectionOperation:
        owner = RuntimeLock(self.root, "supervisor.lock")
        try:
            if read_private(self.pending) != canonical_json(
                operation.model_dump(mode="json")
            ):
                raise ValueError("pending selection changed")
            return self._apply(operation)
        finally:
            owner.close()
