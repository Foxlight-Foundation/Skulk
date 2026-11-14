// Copyright 2026 Foxlight Foundation

use super::{Handler, RpcOut};
use crate::{Behaviour, ConfigBuilder, IdentTopic, MessageAuthenticity, ValidationMode};
use futures::{task::noop_waker_ref, StreamExt};
use libp2p_core::ConnectedPoint;
use libp2p_identity::{Keypair, PeerId};
use libp2p_swarm::{
    behaviour::{ConnectionClosed, ConnectionEstablished, FromSwarm},
    ConnectionHandler, ConnectionHandlerEvent, ConnectionId, NetworkBehaviour,
};
use std::task::{Context, Poll};

const PROTOCOLS: [(&str, Option<&str>); 3] = [
    ("global_events", None),
    ("election_messages", Some("/skulk/election/meshsub")),
    ("telemetry", Some("/skulk/telemetry/meshsub")),
];

fn take_subscription(handler: &mut Handler, topic: &str) -> bool {
    let mut context = Context::from_waker(noop_waker_ref());
    match handler {
        Handler::Enabled(enabled) => matches!(
            enabled.send_queue.poll_next_unpin(&mut context),
            Poll::Ready(Some(RpcOut::Subscribe(seen))) if seen == IdentTopic::new(topic).hash()
        ),
        Handler::Disabled(_) => false,
    }
}

fn overlapping_connections(
    topic: &str,
    prefix: Option<&'static str>,
    closing_takes_replacement: bool,
    blacklisted: bool,
) {
    let mut builder = ConfigBuilder::default();
    builder
        .max_transmit_size(8 * 1024 * 1024)
        .validation_mode(ValidationMode::Strict);
    if let Some(prefix) = prefix {
        builder.protocol_id_prefix(prefix);
    }
    let mut behaviour: Behaviour = Behaviour::new(
        MessageAuthenticity::Signed(Keypair::generate_ed25519()),
        builder.build().unwrap(),
    )
    .unwrap();
    let topic_identity = IdentTopic::new(topic);
    assert!(behaviour.subscribe(&topic_identity).unwrap());
    let peer_id = PeerId::random();
    if blacklisted {
        behaviour.blacklist_peer(&peer_id);
    }
    let first_id = ConnectionId::new_unchecked(1);
    let remaining_id = ConnectionId::new_unchecked(2);
    let endpoint = ConnectedPoint::Listener {
        local_addr: "/ip4/127.0.0.1/tcp/1".parse().unwrap(),
        send_back_addr: "/ip4/127.0.0.1/tcp/2".parse().unwrap(),
    };
    // Drive the actual behaviour/handler events without opening a socket or
    // depending on which connection a scheduler happens to poll first.
    let mut first = behaviour
        .handle_established_inbound_connection(
            first_id,
            peer_id,
            endpoint.get_remote_address(),
            endpoint.get_remote_address(),
        )
        .unwrap();
    behaviour.on_swarm_event(FromSwarm::ConnectionEstablished(ConnectionEstablished {
        peer_id,
        connection_id: first_id,
        endpoint: &endpoint,
        failed_addresses: &[],
        other_established: 0,
    }));
    let mut context = Context::from_waker(noop_waker_ref());
    let first_poll = first.poll(&mut context);
    if !blacklisted {
        assert!(matches!(
            first_poll,
            Poll::Ready(ConnectionHandlerEvent::OutboundSubstreamRequest { .. })
        ));
    }
    let mut remaining = behaviour
        .handle_established_inbound_connection(
            remaining_id,
            peer_id,
            endpoint.get_remote_address(),
            endpoint.get_remote_address(),
        )
        .unwrap();
    behaviour.on_swarm_event(FromSwarm::ConnectionEstablished(ConnectionEstablished {
        peer_id,
        connection_id: remaining_id,
        endpoint: &endpoint,
        failed_addresses: &[],
        other_established: 1,
    }));
    if closing_takes_replacement && !blacklisted {
        assert!(take_subscription(&mut first, topic));
        // Model an in-flight original write followed by prefetch of the new
        // announcement on the same connection that is about to close.
        let _ = first.poll(&mut context);
    }
    assert_eq!(
        take_subscription(&mut remaining, topic),
        !blacklisted && !closing_takes_replacement
    );
    drop(first);
    behaviour.on_swarm_event(FromSwarm::ConnectionClosed(ConnectionClosed {
        peer_id,
        connection_id: first_id,
        endpoint: &endpoint,
        cause: None,
        remaining_established: 1,
    }));
    assert_eq!(take_subscription(&mut remaining, topic), !blacklisted);
    drop(remaining);
    behaviour.on_swarm_event(FromSwarm::ConnectionClosed(ConnectionClosed {
        peer_id,
        connection_id: remaining_id,
        endpoint: &endpoint,
        cause: None,
        remaining_established: 0,
    }));
    assert!(behaviour.all_peers().next().is_none());
    assert!(behaviour
        .topics()
        .any(|local| *local == topic_identity.hash()));
}

#[test]
fn additional_and_surviving_connections_receive_current_subscriptions() {
    for (topic, prefix) in PROTOCOLS {
        overlapping_connections(topic, prefix, false, false);
    }
}

#[test]
fn closure_repairs_an_announcement_prefetched_by_the_dying_handler() {
    for (topic, prefix) in PROTOCOLS {
        overlapping_connections(topic, prefix, true, false);
    }
}

#[test]
fn reannouncement_never_advertises_to_a_blacklisted_peer() {
    for (topic, prefix) in PROTOCOLS {
        overlapping_connections(topic, prefix, false, true);
    }
}
