import hashlib

import pytest

from exo.api.main import API
from exo.shared.election import ElectionMessage
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    SendInputChunk,
    TextGeneration,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import IndexedEvent
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.utils.channels import Receiver, channel


def _build_api() -> tuple[API, Receiver[ForwarderCommand]]:
    command_sender, command_receiver = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    api = API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )
    return api, command_receiver


def _task_params(image: str) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        input=[InputMessage(role="user", content="describe this image")],
        images=[image],
    )


@pytest.mark.asyncio
async def test_text_image_transport_resends_images_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKULK_TEXT_IMAGE_HASH_CACHE", raising=False)
    monkeypatch.delenv("EXO_TEXT_IMAGE_HASH_CACHE", raising=False)
    api, receiver = _build_api()
    image = "aGVsbG8="

    await api._send_text_generation_with_images(_task_params(image))  # pyright: ignore[reportPrivateUsage]
    await api._send_text_generation_with_images(_task_params(image))  # pyright: ignore[reportPrivateUsage]

    messages = await receiver.receive_at_least(4)
    commands = [message.command for message in messages]
    assert isinstance(commands[0], SendInputChunk)
    assert isinstance(commands[1], TextGeneration)
    assert isinstance(commands[2], SendInputChunk)
    assert isinstance(commands[3], TextGeneration)

    first_chunk = commands[0].chunk
    second_chunk = commands[2].chunk
    first_generation = commands[1].task_params
    second_generation = commands[3].task_params
    assert first_chunk.data == image
    assert second_chunk.data == image
    assert first_generation.image_hashes == {}
    assert second_generation.image_hashes == {}
    assert first_generation.total_input_chunks == 1
    assert second_generation.total_input_chunks == 1


@pytest.mark.asyncio
async def test_text_image_hash_cache_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_TEXT_IMAGE_HASH_CACHE", "1")
    api, receiver = _build_api()
    image = "aGVsbG8="
    image_hash = hashlib.sha256(image.encode("ascii")).hexdigest()

    await api._send_text_generation_with_images(_task_params(image))  # pyright: ignore[reportPrivateUsage]
    await api._send_text_generation_with_images(_task_params(image))  # pyright: ignore[reportPrivateUsage]

    messages = await receiver.receive_at_least(3)
    commands = [message.command for message in messages]
    assert isinstance(commands[0], SendInputChunk)
    assert isinstance(commands[1], TextGeneration)
    assert isinstance(commands[2], TextGeneration)
    assert commands[2].task_params.image_hashes == {0: image_hash}
    assert commands[2].task_params.total_input_chunks == 0
