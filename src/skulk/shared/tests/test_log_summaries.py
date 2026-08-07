from skulk.api.types import ImageEditsTaskParams, ImageGenerationTaskParams
from skulk.shared.log_summaries import (
    summarize_command_for_log,
    summarize_task_for_log,
)
from skulk.shared.types import commands as command_types
from skulk.shared.types import tasks as task_types
from skulk.shared.types.audio import (
    AudioTranscriptionTaskParams,
    SpeechSynthesisTaskParams,
)
from skulk.shared.types.chunks import AudioInputChunk
from skulk.shared.types.common import CommandId, ModelId
from skulk.shared.types.embedding import TextEmbeddingTaskParams
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId


def test_serving_log_summaries_exclude_user_payloads() -> None:
    text_params = TextGenerationTaskParams(
        model=ModelId("text-model"),
        input=[InputMessage(role="user", content="TEXT_PAYLOAD_SENTINEL")],
        stream=True,
    )
    image_generation_params = ImageGenerationTaskParams(
        prompt="IMAGE_PROMPT_SENTINEL",
        model="image-model",
    )
    image_edit_params = ImageEditsTaskParams(
        image_data="IMAGE_BYTES_SENTINEL",
        prompt="IMAGE_EDIT_PROMPT_SENTINEL",
        model="image-edit-model",
    )
    embedding_params = TextEmbeddingTaskParams(
        model=ModelId("embedding-model"),
        input_texts=["EMBEDDING_PAYLOAD_SENTINEL"],
    )
    speech_params = SpeechSynthesisTaskParams(
        model=ModelId("tts-model"),
        input_text="SPEECH_TEXT_SENTINEL",
        reference_text="REFERENCE_TEXT_SENTINEL",
        reference_audio_present=True,
        reference_audio_sha256="a" * 64,
        reference_audio_data=b"REFERENCE_AUDIO_SENTINEL",
        seed=42,
    )
    transcription_params = AudioTranscriptionTaskParams(
        model=ModelId("stt-model"),
        audio_sha256="b" * 64,
        audio_data="TRANSCRIPTION_AUDIO_SENTINEL",
        prompt="TRANSCRIPTION_PROMPT_SENTINEL",
        context="TRANSCRIPTION_CONTEXT_SENTINEL",
    )

    commands: list[command_types.Command] = [
        command_types.TextGeneration(
            command_id=CommandId("text-command"),
            task_params=text_params,
        ),
        command_types.ImageGeneration(
            command_id=CommandId("image-command"),
            task_params=image_generation_params,
        ),
        command_types.ImageEdits(
            command_id=CommandId("edit-command"),
            task_params=image_edit_params,
        ),
        command_types.TextEmbedding(
            command_id=CommandId("embedding-command"),
            task_params=embedding_params,
        ),
        command_types.SpeechSynthesis(
            command_id=CommandId("speech-command"),
            task_params=speech_params,
        ),
        command_types.AudioTranscription(
            command_id=CommandId("transcription-command"),
            task_params=transcription_params,
        ),
        command_types.SendInputChunk(
            command_id=CommandId("chunk-command"),
            chunk=AudioInputChunk(
                model=ModelId("stt-model"),
                command_id=CommandId("transcription-command"),
                data="FALLBACK_AUDIO_PAYLOAD_SENTINEL",
                chunk_index=0,
                total_chunks=1,
                filename="FALLBACK_FILENAME_SENTINEL",
                audio_sha256="c" * 64,
            ),
        ),
    ]
    tasks: list[task_types.Task] = [
        task_types.TextGeneration(
            task_id=task_types.TaskId("text-task"),
            command_id=CommandId("text-command"),
            instance_id=InstanceId("text-instance"),
            task_params=text_params,
        ),
        task_types.ImageGeneration(
            task_id=task_types.TaskId("image-task"),
            command_id=CommandId("image-command"),
            instance_id=InstanceId("image-instance"),
            task_params=image_generation_params,
        ),
        task_types.ImageEdits(
            task_id=task_types.TaskId("edit-task"),
            command_id=CommandId("edit-command"),
            instance_id=InstanceId("edit-instance"),
            task_params=image_edit_params,
        ),
        task_types.TextEmbedding(
            task_id=task_types.TaskId("embedding-task"),
            command_id=CommandId("embedding-command"),
            instance_id=InstanceId("embedding-instance"),
            task_params=embedding_params,
        ),
        task_types.SpeechSynthesis(
            task_id=task_types.TaskId("speech-task"),
            command_id=CommandId("speech-command"),
            instance_id=InstanceId("speech-instance"),
            task_params=speech_params,
        ),
        task_types.AudioTranscription(
            task_id=task_types.TaskId("transcription-task"),
            command_id=CommandId("transcription-command"),
            instance_id=InstanceId("transcription-instance"),
            task_params=transcription_params,
        ),
    ]

    summaries = [
        *(summarize_command_for_log(command) for command in commands),
        *(summarize_task_for_log(task) for task in tasks),
    ]
    joined = "\n".join(summaries)

    for sentinel in (
        "TEXT_PAYLOAD_SENTINEL",
        "IMAGE_PROMPT_SENTINEL",
        "IMAGE_BYTES_SENTINEL",
        "IMAGE_EDIT_PROMPT_SENTINEL",
        "EMBEDDING_PAYLOAD_SENTINEL",
        "SPEECH_TEXT_SENTINEL",
        "REFERENCE_TEXT_SENTINEL",
        "REFERENCE_AUDIO_SENTINEL",
        "TRANSCRIPTION_AUDIO_SENTINEL",
        "TRANSCRIPTION_PROMPT_SENTINEL",
        "TRANSCRIPTION_CONTEXT_SENTINEL",
        "FALLBACK_AUDIO_PAYLOAD_SENTINEL",
        "FALLBACK_FILENAME_SENTINEL",
    ):
        assert sentinel not in joined

    assert "input_messages=1" in joined
    assert "prompt_chars=21" in joined
    assert "input_texts=1" in joined
    assert "reference_audio_bytes=24" in joined
    assert joined.count("seed=42") == 2
    assert "audio_payload_chars=28" in joined
    assert "SendInputChunk(command_id='chunk-command')" in joined
