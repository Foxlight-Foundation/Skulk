"""The ``comfy`` video engine: a pinned ComfyUI checkout driven headless.

The worker spawns ComfyUI from the environment provisioned (or pointed at)
for this node, binds a card-described model plus one request onto a graph of
ComfyUI's own MiniMax H3 nodes, submits it over the local HTTP API, follows
progress on the WebSocket, and hands the finished container to the worker
exactly like the test video engine does.
"""
