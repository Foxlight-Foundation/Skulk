from collections.abc import Sequence
from copy import copy
from itertools import count
from math import inf
from os import PathLike
from pathlib import Path
from typing import cast

from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    move_on_after,
    sleep_forever,
)
from filelock import FileLock
from loguru import logger
from pydantic import ValidationError
from skulk_pyo3_bindings import (
    AllQueuesFullError,
    Keypair,
    MessageTooLargeError,
    NetworkingHandle,
    NoPeersSubscribedToTopicError,
    PyFromSwarm,
    ZenohHandle,
)

from skulk.shared.constants import SKULK_NODE_ID_KEYPAIR
from skulk.utils.channels import Receiver, Sender, channel
from skulk.utils.pydantic_ext import CamelCaseModel
from skulk.utils.task_group import TaskGroup

from .connection_message import ConnectionMessage
from .topics import CONNECTION_MESSAGES, DATA, PublishPolicy, TypedTopic


# A significant current limitation of the TopicRouter is that it is not capable
# of preventing feedback, as it does not ask for a system id so cannot tell
# which message is coming/going to which system.
# This is currently only relevant for Election
class TopicRouter[T: CamelCaseModel]:
    def __init__(
        self,
        topic: TypedTopic[T],
        networking_sender: Sender[tuple[str, str | None, bytes]],
        max_buffer_size: float = inf,
    ):
        self.topic: TypedTopic[T] = topic
        self.senders: set[Sender[T]] = set()
        self.origin_senders: set[Sender[tuple[str | None, T]]] = set()
        send, recv = channel[T]()
        self.receiver: Receiver[T] = recv
        self._sender: Sender[T] = send
        self.networking_sender: Sender[tuple[str, str | None, bytes]] = (
            networking_sender
        )

    async def run(self):
        logger.debug(f"Topic Router {self.topic} ready to send")
        with self.receiver as items:
            async for item in items:
                # Check if we should send to network
                if (
                    len(self.senders) == 0
                    and len(self.origin_senders) == 0
                    and self.topic.publish_policy is PublishPolicy.Minimal
                ):
                    await self._send_out(item)
                    continue
                if self.topic.publish_policy is PublishPolicy.Always:
                    await self._send_out(item)
                # Then publish to all senders
                await self.publish(item, origin=None)

    async def shutdown(self):
        logger.debug(f"Shutting down Topic Router {self.topic}")
        # Close all the things!
        for sender in self.senders:
            sender.close()
        for sender in self.origin_senders:
            sender.close()
        self._sender.close()
        self.receiver.close()

    async def publish(self, item: T, origin: str | None = None):
        """
        Publish item T on this topic to all senders.
        NB: this sends to ALL receivers, potentially including receivers held by the object doing the sending.
        You should handle your own output if you hold a sender + receiver pair.
        """
        to_clear: set[Sender[T]] = set()
        for sender in copy(self.senders):
            try:
                await sender.send(item)
            except (ClosedResourceError, BrokenResourceError):
                to_clear.add(sender)
        self.senders -= to_clear

        origin_to_clear: set[Sender[tuple[str | None, T]]] = set()
        for sender in copy(self.origin_senders):
            try:
                await sender.send((origin, item))
            except (ClosedResourceError, BrokenResourceError):
                origin_to_clear.add(sender)
        self.origin_senders -= origin_to_clear

    async def publish_bytes(self, data: bytes, origin: str | None):
        # Wire-format payloads are deserialized strictly (extra="forbid"). During
        # rolling upgrades, an older node may receive a message containing fields
        # it doesn't know about. Catch the validation failure so the gossipsub
        # receive loop survives - dropping the message is recoverable; tearing
        # down the loop is not.
        try:
            item = self.topic.deserialize(data)
        except ValidationError as exception:
            logger.opt(exception=exception).warning(
                f"Dropping malformed or schema-incompatible message on topic "
                f"{self.topic.topic} from {origin}"
            )
            return
        await self.publish(item, origin=origin)

    def new_sender(self) -> Sender[T]:
        return self._sender.clone()

    async def _send_out(self, item: T):
        logger.trace(f"TopicRouter {self.topic.topic} sending {item}")
        # The routing key (Zenoh data plane only) addresses this message to a
        # single subscriber; None broadcasts on the bare topic (#279 Phase 2).
        routing_key = (
            self.topic.routing_key(item)
            if self.topic.routing_key is not None
            else None
        )
        await self.networking_sender.send(
            (str(self.topic.topic), routing_key, self.topic.serialize(item))
        )


class Router:
    @classmethod
    def create(
        cls,
        identity: Keypair,
        bootstrap_peers: Sequence[str] = (),
        listen_port: int = 0,
        zenoh_listen_endpoints: Sequence[str] | None = None,
        zenoh_connect_endpoints: Sequence[str] = (),
        node_id: str | None = None,
    ) -> "Router":
        # When zenoh_listen_endpoints is provided the data plane (DATA topic)
        # rides a Zenoh peer session instead of gossipsub (the zenoh_data_plane
        # flag, decided by the caller). All other topics stay on libp2p.
        zenoh: ZenohHandle | None = None
        if zenoh_listen_endpoints is not None:
            zenoh = ZenohHandle(
                list(zenoh_listen_endpoints), list(zenoh_connect_endpoints)
            )
        # The Zenoh data plane addresses output per owner (key data/<node_id>),
        # so the Router subscribes only to its own id; default to the keypair's
        # node id when the caller doesn't override it.
        resolved_node_id = node_id if node_id is not None else identity.to_node_id()
        return cls(
            handle=NetworkingHandle(identity, list(bootstrap_peers), listen_port),
            zenoh=zenoh,
            node_id=resolved_node_id,
        )

    def __init__(
        self,
        handle: NetworkingHandle,
        zenoh: ZenohHandle | None = None,
        node_id: str = "",
    ):
        self.topic_routers: dict[str, TopicRouter[CamelCaseModel]] = {}
        send, recv = channel[tuple[str, str | None, bytes]]()
        self.networking_receiver: Receiver[tuple[str, str | None, bytes]] = recv
        self._net: NetworkingHandle = handle
        # Optional Zenoh transport for the data plane; None keeps everything on
        # gossipsub (default, until the flag is proven in production).
        self._zenoh: ZenohHandle | None = zenoh
        # This node's id, used as the Zenoh data-plane subscription suffix so a
        # node receives only output addressed to it (#279 Phase 2).
        self._node_id: str = node_id
        self._tmp_networking_sender: Sender[tuple[str, str | None, bytes]] | None = (
            send
        )
        self._id_count = count()
        self._tg: TaskGroup = TaskGroup()

    def uses_zenoh(self, topic: str) -> bool:
        """Whether ``topic`` is routed over the Zenoh data plane (DATA only)."""
        return self._zenoh is not None and topic == DATA.topic

    async def register_topic[T: CamelCaseModel](self, topic: TypedTopic[T]):
        send = self._tmp_networking_sender
        if send:
            self._tmp_networking_sender = None
        else:
            send = self.networking_receiver.clone_sender()
        router = TopicRouter[T](topic, send)
        self.topic_routers[topic.topic] = cast(TopicRouter[CamelCaseModel], router)
        if self._tg.is_running():
            await self._networking_subscribe(topic.topic)

    def sender[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Sender[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts
        assert router is not None
        assert router.topic == topic
        sender = cast(TopicRouter[T], router).new_sender()
        return sender

    def receiver[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Receiver[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts

        assert router is not None
        assert router.topic == topic
        assert router.topic.model_type == topic.model_type

        send, recv = channel[T]()
        router.senders.add(cast(Sender[CamelCaseModel], send))

        return recv

    def receiver_with_origin[T: CamelCaseModel](
        self, topic: TypedTopic[T]
    ) -> Receiver[tuple[str | None, T]]:
        router = self.topic_routers.get(topic.topic, None)
        assert router is not None
        assert router.topic == topic
        assert router.topic.model_type == topic.model_type

        send, recv = channel[tuple[str | None, T]]()
        router.origin_senders.add(
            cast(Sender[tuple[str | None, CamelCaseModel]], send)
        )
        return recv

    async def run(self):
        logger.debug("Starting Router")
        try:
            async with self._tg as tg:
                for topic in self.topic_routers:
                    router = self.topic_routers[topic]
                    tg.start_soon(router.run)
                tg.start_soon(self._networking_recv)
                tg.start_soon(self._networking_publish)
                if self._zenoh is not None:
                    tg.start_soon(self._zenoh_recv)
                # subscribe to pending topics
                for topic in self.topic_routers:
                    await self._networking_subscribe(topic)
                # Router only shuts down if you cancel it.
                await sleep_forever()
        finally:
            with move_on_after(1, shield=True):
                for topic in self.topic_routers:
                    await self._networking_unsubscribe(str(topic))

    async def shutdown(self):
        logger.debug("Shutting down Router")
        self._tg.cancel_tasks()

    async def _networking_subscribe(self, topic: str):
        if self.uses_zenoh(topic):
            assert self._zenoh is not None
            # Subscribe only to output addressed to this node (data/<node_id>),
            # not the whole topic, so the serving worker's unicast reaches just
            # the owning API node instead of fanning out to every node (#279
            # Phase 2). The owner is keyed by node id; the bare topic is never
            # subscribed, so non-owners receive nothing.
            key = f"{topic}/{self._node_id}"
            await self._zenoh.zenoh_subscribe(key)
            logger.info(f"Subscribed to {key} (zenoh data plane)")
            return
        await self._net.gossipsub_subscribe(topic)
        logger.info(f"Subscribed to {topic}")

    async def _networking_unsubscribe(self, topic: str):
        if self.uses_zenoh(topic):
            # Zenoh subscribers are undeclared when the session closes; there is
            # no per-topic unsubscribe to issue here.
            return
        await self._net.gossipsub_unsubscribe(topic)
        logger.info(f"Unsubscribed from {topic}")

    async def _zenoh_recv(self):
        """Drain inbound DATA-plane samples from Zenoh into their topic router.

        Parallel to :meth:`_networking_recv` (which handles the gossipsub
        planes). Demux is by the ``command_id`` inside the payload, done
        downstream in the DATA topic's consumer, exactly as on gossipsub.
        """
        assert self._zenoh is not None
        try:
            while True:
                message = await self._zenoh.recv()
                # The sample key is data/<owner_node>; the TopicRouter is keyed by
                # the bare topic, so strip the routing suffix to find it (#279
                # Phase 2).
                topic = message.topic.split("/", 1)[0]
                if topic not in self.topic_routers:
                    logger.warning(
                        f"Received zenoh message on unknown or inactive topic "
                        f"{message.topic}"
                    )
                    continue
                # No origin peer id on the zenoh data plane; the data plane does
                # not use origin (output chunks never mutate State).
                await self.topic_routers[topic].publish_bytes(message.data, None)
        except Exception as exception:
            logger.opt(exception=exception).error(
                "Zenoh data-plane receive loop terminated unexpectedly"
            )
            raise

    async def _networking_recv(self):
        try:
            while True:
                from_swarm = await self._net.recv()
                logger.debug(from_swarm)
                match from_swarm:
                    case PyFromSwarm.Message(origin, topic, data):
                        logger.trace(
                            f"Received message on {topic} from {origin} with payload {data}"
                        )
                        if topic not in self.topic_routers:
                            logger.warning(
                                f"Received message on unknown or inactive topic {topic}"
                            )
                            continue
                        router = self.topic_routers[topic]
                        await router.publish_bytes(data, origin)
                    case PyFromSwarm.Connection():
                        message = ConnectionMessage.from_update(from_swarm)
                        logger.trace(
                            f"Received message on connection_messages with payload {message}"
                        )
                        if CONNECTION_MESSAGES.topic in self.topic_routers:
                            router = self.topic_routers[CONNECTION_MESSAGES.topic]
                            assert router.topic.model_type == ConnectionMessage
                            router = cast(TopicRouter[ConnectionMessage], router)
                            await router.publish(message)
                    case _:
                        logger.critical(
                            "failed to exhaustively check FromSwarm messages - logic error"
                        )
        except Exception as exception:
            logger.opt(exception=exception).error(
                "Gossipsub receive loop terminated unexpectedly"
            )
            raise

    async def _networking_publish(self):
        with self.networking_receiver as networked_items:
            async for topic, routing_key, data in networked_items:
                try:
                    logger.trace(f"Sending message on {topic} with payload {data}")
                    if len(data) > 1024 * 1024:
                        logger.warning(
                            "Sending overlarge payload, network performance may be temporarily degraded"
                        )
                    if self.uses_zenoh(topic):
                        assert self._zenoh is not None
                        # Address the chunk to its owning API node (data/<owner>).
                        # Nodes subscribe only to data/<own_node_id>, never the
                        # bare topic, so a message with no routing key reaches no
                        # subscriber. That should not happen - every serving task
                        # carries owner_node (#279 Phase 2) - so warn loudly
                        # rather than dropping it silently if it ever does (#310
                        # review).
                        if not routing_key:
                            logger.warning(
                                f"Zenoh DATA publish on {topic} has no routing key "
                                f"(owner_node unset); no node subscribes to the bare "
                                f"topic, so this output would be lost. Dropping a "
                                f"chunk of {len(data)} bytes."
                            )
                            continue
                        await self._zenoh.zenoh_publish(f"{topic}/{routing_key}", data)
                        continue
                    await self._net.gossipsub_publish(topic, data)
                except NoPeersSubscribedToTopicError:
                    pass
                except AllQueuesFullError:
                    logger.warning(f"All peer queues full, dropping message on {topic}")
                except MessageTooLargeError:
                    logger.warning(
                        f"Message too large for gossipsub on {topic} ({len(data)} bytes), dropping"
                    )


def get_node_id_keypair(
    path: str | bytes | PathLike[str] | PathLike[bytes] = SKULK_NODE_ID_KEYPAIR,
) -> Keypair:
    """
    Obtains the :class:`Keypair` associated with this node-ID.
    Obtain the :class:`PeerId` by from it.
    """
    # TODO(evan): bring back node id persistence once we figure out how to deal with duplicates
    return Keypair.generate()

    def lock_path(path: str | bytes | PathLike[str] | PathLike[bytes]) -> Path:
        return Path(str(path) + ".lock")

    # operate with cross-process lock to avoid race conditions
    with FileLock(lock_path(path)):
        with open(path, "a+b") as f:  # opens in append-mode => starts at EOF
            # if non-zero EOF, then file exists => use to get node-ID
            if f.tell() != 0:
                f.seek(0)  # go to start & read protobuf-encoded bytes
                protobuf_encoded = f.read()

                try:  # if decoded successfully, save & return
                    return Keypair.from_bytes(protobuf_encoded)
                except ValueError as e:  # on runtime error, assume corrupt file
                    logger.warning(f"Encountered error when trying to get keypair: {e}")

        # if no valid credentials, create new ones and persist
        with open(path, "w+b") as f:
            keypair = Keypair.generate()
            f.write(keypair.to_bytes())
            return keypair
