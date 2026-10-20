from . import _io as io

class ComfyExtension:
    async def get_node_list(self) -> list[type[io.ComfyNode]]: ...
