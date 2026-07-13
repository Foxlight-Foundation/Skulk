use futures_lite::StreamExt;
use networking::swarm::{FromSwarm, ToSwarm, create_swarm};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

/// Helper: find a free TCP port.
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Two nodes connect via bootstrap peers — no mDNS needed.
///
/// Node A listens on a fixed port. Node B bootstraps to A's address.
/// We verify that B emits `FromSwarm::Discovered` for A's peer ID.
#[tokio::test]
async fn two_nodes_connect_via_bootstrap_peers() {
    let port_a = free_port();

    // Node A: listens on a known port, no bootstrap peers
    let keypair_a = libp2p::identity::Keypair::generate_ed25519();
    let peer_id_a = keypair_a.public().to_peer_id();
    let (_tx_a, rx_a) = mpsc::channel(16);
    let swarm_a = create_swarm(keypair_a, rx_a, vec![], port_a).expect("create swarm A");
    let mut stream_a = swarm_a.into_stream();

    // Node B: bootstraps to A's address
    let keypair_b = libp2p::identity::Keypair::generate_ed25519();
    let (_tx_b, rx_b) = mpsc::channel(16);
    let swarm_b = create_swarm(
        keypair_b,
        rx_b,
        vec![format!("/ip4/127.0.0.1/tcp/{port_a}")],
        0,
    )
    .expect("create swarm B");
    let mut stream_b = swarm_b.into_stream();

    // Wait for B to discover A (connection established)
    let connected = timeout(Duration::from_secs(10), async {
        loop {
            tokio::select! {
                Some(event) = stream_a.next() => {
                    // A will also see B connect, but we check from B's perspective
                    let _ = event;
                }
                Some(event) = stream_b.next() => {
                    if let FromSwarm::Discovered { peer_id } = event {
                        if peer_id == peer_id_a {
                            return true;
                        }
                    }
                }
            }
        }
    })
    .await;

    assert!(
        connected.is_ok() && connected.unwrap(),
        "Node B should discover Node A via bootstrap peer"
    );
}

/// The isolated election gossipsub protocol negotiates and carries messages.
#[tokio::test]
async fn two_nodes_exchange_election_messages_on_isolated_protocol() {
    let port_a = free_port();
    let keypair_a = libp2p::identity::Keypair::generate_ed25519();
    let peer_id_a = keypair_a.public().to_peer_id();
    let (tx_a, rx_a) = mpsc::channel(16);
    let swarm_a = create_swarm(keypair_a, rx_a, vec![], port_a).expect("create swarm A");

    let keypair_b = libp2p::identity::Keypair::generate_ed25519();
    let (tx_b, rx_b) = mpsc::channel(16);
    let swarm_b = create_swarm(
        keypair_b,
        rx_b,
        vec![format!("/ip4/127.0.0.1/tcp/{port_a}")],
        0,
    )
    .expect("create swarm B");

    let (events_a_tx, mut events_a_rx) = mpsc::channel(32);
    let (events_b_tx, mut events_b_rx) = mpsc::channel(32);
    tokio::spawn(async move {
        let mut stream = swarm_a.into_stream();
        while let Some(event) = stream.next().await {
            if events_a_tx.send(event).await.is_err() {
                return;
            }
        }
    });
    tokio::spawn(async move {
        let mut stream = swarm_b.into_stream();
        while let Some(event) = stream.next().await {
            if events_b_tx.send(event).await.is_err() {
                return;
            }
        }
    });

    timeout(Duration::from_secs(10), async {
        loop {
            tokio::select! {
                Some(_) = events_a_rx.recv() => {}
                Some(event) = events_b_rx.recv() => {
                    if matches!(event, FromSwarm::Discovered { peer_id } if peer_id == peer_id_a) {
                        return;
                    }
                }
            }
        }
    })
    .await
    .expect("B should discover A");

    let topic = "election_messages".to_string();
    let (subscribe_a_tx, subscribe_a_rx) = oneshot::channel();
    tx_a.send(ToSwarm::Subscribe {
        topic: topic.clone(),
        result_sender: subscribe_a_tx,
    })
    .await
    .expect("send A subscription");
    let (subscribe_b_tx, subscribe_b_rx) = oneshot::channel();
    tx_b.send(ToSwarm::Subscribe {
        topic: topic.clone(),
        result_sender: subscribe_b_tx,
    })
    .await
    .expect("send B subscription");
    assert!(
        subscribe_a_rx
            .await
            .expect("A subscription response")
            .expect("A subscription succeeds")
    );
    assert!(
        subscribe_b_rx
            .await
            .expect("B subscription response")
            .expect("B subscription succeeds")
    );

    // Allow gossipsub subscription propagation before publishing.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let payload = b"isolated-election".to_vec();
    let (publish_tx, publish_rx) = oneshot::channel();
    tx_a.send(ToSwarm::Publish {
        topic: topic.clone(),
        data: payload.clone(),
        result_sender: publish_tx,
    })
    .await
    .expect("send election publish");
    publish_rx
        .await
        .expect("publish response")
        .expect("election publish succeeds");

    timeout(Duration::from_secs(10), async {
        let mut matching_deliveries = 0;
        while let Some(event) = events_b_rx.recv().await {
            if matches!(
                event,
                FromSwarm::Message { topic: ref received_topic, data: ref received_data, .. }
                    if received_topic == &topic && received_data == &payload
            ) {
                matching_deliveries += 1;
                if matching_deliveries == 2 {
                    return;
                }
            }
        }
        panic!("B event stream closed before dual-path election delivery");
    })
    .await
    .expect("B should receive isolated and rolling-compatibility election copies");
}

/// Empty bootstrap peers should work (backward compatible).
#[tokio::test]
async fn create_swarm_with_empty_bootstrap_peers() {
    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let (_tx, rx) = mpsc::channel(16);
    let swarm = create_swarm(keypair, rx, vec![], 0);
    assert!(
        swarm.is_ok(),
        "create_swarm with no bootstrap peers should succeed"
    );
}

/// Invalid multiaddr strings are silently filtered out.
#[tokio::test]
async fn create_swarm_ignores_invalid_bootstrap_addrs() {
    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let (_tx, rx) = mpsc::channel(16);
    let swarm = create_swarm(
        keypair,
        rx,
        vec![
            "not-a-valid-multiaddr".to_string(),
            "".to_string(),
            "/ip4/10.0.0.1/tcp/30000".to_string(), // valid
        ],
        0,
    );
    assert!(
        swarm.is_ok(),
        "create_swarm should succeed even with invalid bootstrap addrs"
    );
}

/// Fixed listen port works correctly.
#[tokio::test]
async fn create_swarm_with_fixed_port() {
    let port = free_port();
    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let (_tx, rx) = mpsc::channel(16);
    let swarm = create_swarm(keypair, rx, vec![], port);
    assert!(swarm.is_ok(), "create_swarm with fixed port should succeed");
}
