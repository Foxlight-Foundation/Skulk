//! Zenoh peer session for the Skulk data plane (Phase 1, flag-gated).
//!
//! This runs ALONGSIDE the libp2p [`crate::swarm::Swarm`]: control, telemetry,
//! and election stay on gossipsub; only the DATA topic (per-token output) is
//! routed here when the `zenoh_data_plane` flag is on. The session is a Zenoh
//! `peer` with **multicast scouting disabled** and gossip + explicit endpoints,
//! which is the posture proven (Phase 0) to form a mesh on the macOS fleet over
//! both the LAN and the Thunderbolt chain without tripping Local Network
//! Privacy.
//!
//! Ordering discipline: publishers are declared `Reliable` with
//! `CongestionControl::Block` on a single fixed `Priority`, so a single
//! publisher's samples on one key are delivered FIFO — the property that lets
//! Phase 3 delete the app-layer reorder buffer.

use std::collections::HashMap;

use tokio::sync::Mutex;
use tokio::sync::mpsc;
use zenoh::Session;
use zenoh::pubsub::{Publisher, Subscriber};
use zenoh::qos::{CongestionControl, Priority, Reliability};

use crate::alias::{AnyError, AnyResult};

/// Endpoint configuration for the Zenoh peer session.
#[derive(Debug, Clone, Default)]
pub struct ZenohConfig {
    /// Local endpoints to listen on, e.g. `tcp/0.0.0.0:7447`.
    pub listen_endpoints: Vec<String>,
    /// Peer endpoints to connect to (explicit, since multicast is off).
    pub connect_endpoints: Vec<String>,
}

fn json_str_array(items: &[String]) -> String {
    // JSON5 array of quoted strings; endpoints are operator-controlled config.
    let quoted: Vec<String> = items.iter().map(|e| format!("{e:?}")).collect();
    format!("[{}]", quoted.join(","))
}

/// A live Zenoh peer session plus the publishers/subscribers declared on it.
///
/// `recv` pulls the next inbound `(topic, payload)` delivered to any declared
/// subscriber, demuxed by the caller (the Python data-plane consumer keys on the
/// `command_id` carried inside the payload, exactly as the gossipsub path does).
pub struct ZenohSession {
    session: Session,
    publishers: Mutex<HashMap<String, Publisher<'static>>>,
    subscribers: Mutex<HashMap<String, Subscriber<()>>>,
    inbound_tx: mpsc::UnboundedSender<(String, Vec<u8>)>,
    inbound_rx: Mutex<mpsc::UnboundedReceiver<(String, Vec<u8>)>>,
}

impl ZenohSession {
    /// Open a Zenoh `peer` session with multicast off, gossip on, and the
    /// supplied explicit endpoints.
    pub async fn open(config: ZenohConfig) -> AnyResult<Self> {
        let mut zconfig = zenoh::Config::default();
        let set = |c: &mut zenoh::Config, key: &str, val: &str| -> AnyResult<()> {
            c.insert_json5(key, val)
                .map_err(|e| -> AnyError { format!("zenoh config {key}: {e}").into() })
        };
        set(&mut zconfig, "mode", "\"peer\"")?;
        set(&mut zconfig, "scouting/multicast/enabled", "false")?;
        set(&mut zconfig, "scouting/gossip/enabled", "true")?;
        if !config.listen_endpoints.is_empty() {
            set(
                &mut zconfig,
                "listen/endpoints",
                &json_str_array(&config.listen_endpoints),
            )?;
        }
        if !config.connect_endpoints.is_empty() {
            set(
                &mut zconfig,
                "connect/endpoints",
                &json_str_array(&config.connect_endpoints),
            )?;
        }

        let session = zenoh::open(zconfig)
            .await
            .map_err(|e| -> AnyError { format!("zenoh open: {e}").into() })?;
        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
        Ok(Self {
            session,
            publishers: Mutex::new(HashMap::new()),
            subscribers: Mutex::new(HashMap::new()),
            inbound_tx,
            inbound_rx: Mutex::new(inbound_rx),
        })
    }

    /// Publish `data` on `topic` (Reliable + Block + single fixed priority).
    ///
    /// The publisher for a topic is declared once and reused, preserving the
    /// single-publisher-per-key FIFO ordering the data plane depends on.
    pub async fn publish(&self, topic: &str, data: Vec<u8>) -> AnyResult<()> {
        let mut publishers = self.publishers.lock().await;
        if !publishers.contains_key(topic) {
            let publisher = self
                .session
                .declare_publisher(topic.to_string())
                .congestion_control(CongestionControl::Block)
                .priority(Priority::Data)
                .reliability(Reliability::Reliable)
                .await
                .map_err(|e| -> AnyError { format!("declare_publisher {topic}: {e}").into() })?;
            publishers.insert(topic.to_string(), publisher);
        }
        let publisher = publishers
            .get(topic)
            .expect("publisher just inserted for topic");
        publisher
            .put(data)
            .await
            .map_err(|e| -> AnyError { format!("publish {topic}: {e}").into() })
    }

    /// Declare a subscriber on `topic`; inbound samples are forwarded to `recv`.
    ///
    /// Idempotent: subscribing to an already-subscribed topic is a no-op.
    pub async fn subscribe(&self, topic: &str) -> AnyResult<()> {
        let mut subscribers = self.subscribers.lock().await;
        if subscribers.contains_key(topic) {
            return Ok(());
        }
        let tx = self.inbound_tx.clone();
        let subscriber = self
            .session
            .declare_subscriber(topic.to_string())
            .callback(move |sample| {
                let key = sample.key_expr().as_str().to_string();
                let payload = sample.payload().to_bytes().to_vec();
                // The receiver is dropped only on session teardown; ignore the
                // closed-channel error during shutdown.
                let _ = tx.send((key, payload));
            })
            .await
            .map_err(|e| -> AnyError { format!("declare_subscriber {topic}: {e}").into() })?;
        subscribers.insert(topic.to_string(), subscriber);
        Ok(())
    }

    /// Await the next inbound `(topic, payload)`, or `None` once the session is
    /// closed and all senders are dropped.
    pub async fn recv(&self) -> Option<(String, Vec<u8>)> {
        let mut rx = self.inbound_rx.lock().await;
        rx.recv().await
    }
}
