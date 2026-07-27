"""Payload-free summaries for serving commands and worker tasks."""

from skulk.shared.types import commands as command_types
from skulk.shared.types import tasks as task_types


def summarize_command_for_log(command: command_types.Command) -> str:
    """Return an operational serving-command summary without user payloads."""

    if isinstance(command, command_types.TextGeneration):
        params = command.task_params
        return (
            "TextGeneration("
            f"command_id={command.command_id!r}, "
            f"model={params.model!r}, "
            f"input_messages={len(params.input)}, "
            f"chat_template_messages={len(params.chat_template_messages or [])}, "
            f"images={len(params.images)}, "
            f"cached_image_indices={sorted(params.image_hashes.keys())}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"image_count={params.image_count}, "
            f"stream={params.stream})"
        )
    if isinstance(command, command_types.ImageGeneration):
        params = command.task_params
        return (
            "ImageGeneration("
            f"command_id={command.command_id!r}, "
            f"model={params.model!r}, "
            f"prompt_chars={len(params.prompt)}, "
            f"n={params.n!r}, "
            f"size={params.size!r}, "
            f"stream={params.stream!r})"
        )
    if isinstance(command, command_types.ImageEdits):
        params = command.task_params
        return (
            "ImageEdits("
            f"command_id={command.command_id!r}, "
            f"model={params.model!r}, "
            f"prompt_chars={len(params.prompt)}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"has_inline_image_data={bool(params.image_data)}, "
            f"n={params.n!r}, "
            f"size={params.size!r}, "
            f"stream={params.stream!r})"
        )
    if isinstance(command, command_types.TextEmbedding):
        params = command.task_params
        return (
            "TextEmbedding("
            f"command_id={command.command_id!r}, "
            f"model={params.model!r}, "
            f"input_texts={len(params.input_texts)}, "
            f"encoding_format={params.encoding_format!r})"
        )
    if isinstance(command, command_types.SpeechSynthesis):
        params = command.task_params
        return (
            "SpeechSynthesis("
            f"command_id={command.command_id!r}, "
            f"model={params.model!r}, "
            f"input_chars={len(params.input_text)}, "
            f"response_format={params.response_format!r}, "
            f"reference_audio_present={params.reference_audio_present}, "
            f"reference_audio_bytes={len(params.reference_audio_data or b'')}, "
            f"stream={params.stream})"
        )
    if isinstance(command, command_types.AudioTranscription):
        params = command.task_params
        return (
            "AudioTranscription("
            f"command_id={command.command_id!r}, "
            f"model={params.model!r}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"audio_payload_chars={len(params.audio_data or '')}, "
            f"translate_to_english={params.translate_to_english}, "
            f"stream={params.stream})"
        )
    if isinstance(command, command_types.RealtimeAudioTranscription):
        params = command.task_params
        return (
            "RealtimeAudioTranscription("
            f"command_id={command.command_id!r}, "
            f"model={params.model!r}, "
            f"input_sample_rate={params.input_sample_rate})"
        )
    return (
        f"{command.__class__.__name__}("
        f"command_id={command.command_id!r})"
    )


def summarize_task_for_log(task: task_types.Task) -> str:
    """Return an operational worker-task summary without user payloads."""

    if isinstance(task, task_types.CreateRunner):
        shard = task.bound_instance.bound_shard
        return (
            "CreateRunner("
            f"instance_id={task.instance_id!r}, "
            f"runner_id={task.bound_instance.bound_runner_id!r}, "
            f"node_id={task.bound_instance.bound_node_id!r}, "
            f"device_rank={shard.device_rank}, "
            f"world_size={shard.world_size}, "
            f"layers={shard.start_layer}:{shard.end_layer})"
        )
    if isinstance(task, task_types.DownloadModel):
        shard = task.shard_metadata
        return (
            "DownloadModel("
            f"instance_id={task.instance_id!r}, "
            f"model={shard.model_card.model_id!r}, "
            f"device_rank={shard.device_rank}, "
            f"world_size={shard.world_size}, "
            f"layers={shard.start_layer}:{shard.end_layer})"
        )
    if isinstance(task, task_types.Shutdown):
        return (
            f"Shutdown(instance_id={task.instance_id!r}, runner_id={task.runner_id!r})"
        )
    if isinstance(task, task_types.LoadModel):
        return f"LoadModel(instance_id={task.instance_id!r})"
    if isinstance(task, task_types.StartWarmup):
        return f"StartWarmup(instance_id={task.instance_id!r})"
    if isinstance(task, task_types.CancelTask):
        return (
            "CancelTask("
            f"instance_id={task.instance_id!r}, "
            f"runner_id={task.runner_id!r}, "
            f"cancelled_task_id={task.cancelled_task_id!r})"
        )
    if isinstance(task, task_types.TextGeneration):
        params = task.task_params
        return (
            "TextGeneration("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"instance_id={task.instance_id!r}, "
            f"model={params.model!r}, "
            f"input_messages={len(params.input)}, "
            f"chat_template_messages={len(params.chat_template_messages or [])}, "
            f"images={len(params.images)}, "
            f"cached_image_indices={sorted(params.image_hashes.keys())}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"image_count={params.image_count}, "
            f"stream={params.stream}, "
            f"reasoning_effort={params.reasoning_effort!r}, "
            f"enable_thinking={params.enable_thinking!r})"
        )
    if isinstance(task, task_types.ImageGeneration):
        params = task.task_params
        return (
            "ImageGeneration("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"instance_id={task.instance_id!r}, "
            f"model={params.model!r}, "
            f"prompt_chars={len(params.prompt)}, "
            f"n={params.n!r}, "
            f"size={params.size!r}, "
            f"stream={params.stream!r})"
        )
    if isinstance(task, task_types.ImageEdits):
        params = task.task_params
        return (
            "ImageEdits("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"instance_id={task.instance_id!r}, "
            f"model={params.model!r}, "
            f"prompt_chars={len(params.prompt)}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"has_inline_image_data={bool(params.image_data)}, "
            f"n={params.n!r}, "
            f"size={params.size!r}, "
            f"stream={params.stream!r})"
        )
    if isinstance(task, task_types.TextEmbedding):
        params = task.task_params
        return (
            "TextEmbedding("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"instance_id={task.instance_id!r}, "
            f"model={params.model!r}, "
            f"input_texts={len(params.input_texts)}, "
            f"encoding_format={params.encoding_format!r})"
        )
    if isinstance(task, task_types.SpeechSynthesis):
        params = task.task_params
        return (
            "SpeechSynthesis("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"instance_id={task.instance_id!r}, "
            f"model={params.model!r}, "
            f"input_chars={len(params.input_text)}, "
            f"response_format={params.response_format!r}, "
            f"reference_audio_present={params.reference_audio_present}, "
            f"reference_audio_bytes={len(params.reference_audio_data or b'')}, "
            f"stream={params.stream})"
        )
    if isinstance(task, task_types.AudioTranscription):
        params = task.task_params
        return (
            "AudioTranscription("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"instance_id={task.instance_id!r}, "
            f"model={params.model!r}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"audio_payload_chars={len(params.audio_data or '')}, "
            f"translate_to_english={params.translate_to_english}, "
            f"stream={params.stream})"
        )
    if isinstance(task, task_types.RealtimeAudioTranscription):
        params = task.task_params
        return (
            "RealtimeAudioTranscription("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"instance_id={task.instance_id!r}, "
            f"model={params.model!r}, "
            f"input_sample_rate={params.input_sample_rate})"
        )
    return (
        f"{task.__class__.__name__}("
        f"task_id={task.task_id!r}, "
        f"instance_id={task.instance_id!r}, "
        f"task_status={task.task_status!r})"
    )
