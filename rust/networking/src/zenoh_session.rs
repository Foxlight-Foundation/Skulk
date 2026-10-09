//! Zenoh peer session for the Skulk data plane.
//!
//! This runs ALONGSIDE the libp2p [`crate::swarm::Swarm`]: control, telemetry,
//! and election stay on gossipsub; the node-addressed data families are routed
//! here when the Zenoh data plane is on. The session is a Zenoh `peer` with
//! gossip discovery. Explicitly configured fleets use fixed connect endpoints
//! with multicast scouting disabled. A zero-config installation uses multicast
//! scouting so local peers discover each other without a fleet-specific
//! endpoint list.
//!
//! Cluster boundary: the session is scoped and authenticated by the same
//! cluster key as the libp2p control plane (see `data_plane_trust.rs`).
//! Multicast scouting uses a port derived from that key, every link is mutual
//! TLS chained to the key-derived authority (`tls/` is the only link protocol
//! the session opens or accepts), and an access-control policy admits traffic
//! only from links whose peer certificate carries the cluster's member name.
//!
//! Ordering discipline: publishers are declared `Reliable` with
//! `CongestionControl::Block` on a single fixed `Priority`, so a single
//! publisher's samples on one key are delivered FIFO — the property that lets
//! Phase 3 delete the app-layer reorder buffer.

use std::collections::HashMap;
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use zenoh::Session;
use zenoh::pubsub::{Publisher, Subscriber};
use zenoh::qos::{CongestionControl, Priority, Reliability};
use zenoh::session::ZenohId;
use zeroize::Zeroizing;

use crate::alias::{AnyError, AnyResult};
use crate::data_plane_trust::DataPlaneTrust;

/// Bound on buffered inbound samples awaiting the Python consumer. The Zenoh
/// subscribe callback is non-blocking and drops on a full channel rather than
/// growing memory without limit if the consumer falls behind (DATA is
/// best-effort; the API reorder buffer + idle backstop tolerate gaps, the same
/// as the gossipsub path's queue-full drop).
const INBOUND_BUFFER_CAPACITY: usize = 4096;

/// The only link protocol the data plane opens or accepts.
const DATA_PLANE_ENDPOINT_PREFIX: &str = "tls/";

/// Every message kind Zenoh's access control can filter. The membership policy
/// allows all of them for cluster members and, by the default-deny rule, none
/// of them for any other link.
const ACCESS_CONTROL_MESSAGES: &str = r#"["put", "delete", "declare_subscriber", "query", "declare_queryable", "reply", "liveliness_token", "declare_liveliness_subscriber", "liveliness_query"]"#;

/// Endpoint configuration for the Zenoh peer session.
#[derive(Debug, Clone, Default)]
pub struct ZenohConfig {
    /// Local endpoints to listen on, e.g. `tls/192.168.1.10:7447`. Only `tls/`
    /// endpoints are accepted.
    pub listen_endpoints: Vec<String>,
    /// Peer endpoints to connect to in an explicitly routed deployment. Only
    /// `tls/` endpoints are accepted.
    pub connect_endpoints: Vec<String>,
    /// Whether local multicast scouting discovers peers when no explicit
    /// fleet endpoint list is available. Scouting always uses the port
    /// derived from the cluster key, never Zenoh's shared default.
    pub multicast_scouting: bool,
    /// Optional session namespace (#308): a non-wildcard key-expr prefix that
    /// Zenoh transparently prepends to every published/subscribed key. The
    /// cluster boundary itself is the key-derived mutual TLS and access
    /// control; the prefix additionally keeps two clusters apart in routing if
    /// their scouting ports collide. The caller must pass a valid,
    /// already-sanitized key-expr segment (derived from
    /// `SKULK_LIBP2P_NAMESPACE`); `None` leaves keys unprefixed (legacy).
    pub namespace: Option<String>,
}

fn json_str_array(items: &[String]) -> String {
    // JSON5 array of quoted strings; endpoints are operator-controlled config.
    let quoted: Vec<String> = items.iter().map(|e| format!("{e:?}")).collect();
    format!("[{}]", quoted.join(","))
}

fn set(config: &mut zenoh::Config, key: &str, value: &str) -> AnyResult<()> {
    config
        .insert_json5(key, value)
        .map_err(|e| -> AnyError { format!("zenoh config {key}: {e}").into() })
}

/// Like [`set`], but the error never echoes the value: used for TLS material.
fn set_secret(config: &mut zenoh::Config, key: &str, value: &str) -> AnyResult<()> {
    config
        .insert_json5(key, value)
        .map_err(|_| -> AnyError { format!("zenoh config {key}: value rejected").into() })
}

/// Replace every occurrence of a secret in an error message before it can
/// reach Python and its logs.
fn redact(message: String, secrets: &[Zeroizing<String>]) -> String {
    secrets
        .iter()
        .filter(|secret| !secret.is_empty())
        .fold(message, |message, secret| {
            message.replace(secret.as_str(), "<redacted>")
        })
}

/// The access-control policy: deny by default, allow everything on links whose
/// peer certificate presents the cluster's member name.
fn access_control_policy(member_name: &str) -> String {
    format!(
        r#"{{
            enabled: true,
            default_permission: "deny",
            rules: [{{
                id: "skulk-cluster-traffic",
                permission: "allow",
                flows: ["egress", "ingress"],
                messages: {ACCESS_CONTROL_MESSAGES},
                key_exprs: ["**"],
            }}],
            subjects: [{{ id: "skulk-cluster-members", cert_common_names: [{member_name:?}] }}],
            policies: [{{
                id: "skulk-cluster-policy",
                rules: ["skulk-cluster-traffic"],
                subjects: ["skulk-cluster-members"],
            }}],
        }}"#
    )
}

/// Whether a session enforces the membership policy. Production sessions always
/// do; the tests build one peer without it to model an intruder that ignores
/// policy.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Membership {
    Enforced,
    #[cfg(test)]
    Unenforced,
}

/// Build the Zenoh configuration for one data-plane session.
///
/// Returns the configuration plus the secret strings it embeds, so a failed
/// open can redact them from its error.
fn session_config(
    config: &ZenohConfig,
    trust: &DataPlaneTrust,
    membership: Membership,
) -> AnyResult<(zenoh::Config, Vec<Zeroizing<String>>)> {
    for endpoint in config
        .listen_endpoints
        .iter()
        .chain(&config.connect_endpoints)
    {
        if !endpoint.starts_with(DATA_PLANE_ENDPOINT_PREFIX) {
            return Err(format!(
                "zenoh data-plane endpoint {endpoint:?} must use the {DATA_PLANE_ENDPOINT_PREFIX} transport"
            )
            .into());
        }
    }

    let mut zconfig = zenoh::Config::default();
    set(&mut zconfig, "mode", "\"peer\"")?;
    set(
        &mut zconfig,
        "scouting/multicast/enabled",
        if config.multicast_scouting {
            "true"
        } else {
            "false"
        },
    )?;
    if config.multicast_scouting {
        // Cluster-scoped discovery: clusters with different keys scout on
        // different ports of the shared group and never meet.
        set(
            &mut zconfig,
            "scouting/multicast/address",
            &format!("{:?}", trust.scouting_address()),
        )?;
    }
    set(&mut zconfig, "scouting/gossip/enabled", "true")?;
    // Namespace isolation (#308): Zenoh transparently prefixes every key
    // with this non-wildcard key-expr. Authentication now comes from TLS and
    // access control; the prefix keeps keys tidy and separates any two
    // clusters whose scouting ports happen to collide.
    if let Some(ns) = &config.namespace {
        set(&mut zconfig, "namespace", &format!("{ns:?}"))?;
    }
    // Always set both lists, even when empty: Zenoh's peer default listener is
    // `tcp/[::]:0` (every interface, plaintext), which the TLS-only protocol
    // set below would refuse anyway.
    set(
        &mut zconfig,
        "listen/endpoints",
        &json_str_array(&config.listen_endpoints),
    )?;
    set(
        &mut zconfig,
        "connect/endpoints",
        &json_str_array(&config.connect_endpoints),
    )?;

    // TLS is the only link protocol: scouted or gossiped `tcp/` locators are
    // never dialed and no plaintext listener can exist.
    set(&mut zconfig, "transport/link/protocols", r#"["tls"]"#)?;
    // One link per peer. The access-control subject of a transport is fixed
    // when its first link forms, so no further link may ever join it.
    set(&mut zconfig, "transport/unicast/max_links", "1")?;

    let authority = BASE64.encode(trust.authority_certificate_pem().as_bytes());
    let certificate = BASE64.encode(trust.member_certificate_pem().as_bytes());
    let private_key = Zeroizing::new(BASE64.encode(trust.member_private_key_pem().as_bytes()));
    let quoted_private_key = Zeroizing::new(format!("\"{}\"", private_key.as_str()));
    set(
        &mut zconfig,
        "transport/link/tls/root_ca_certificate_base64",
        &format!("\"{authority}\""),
    )?;
    for role in ["listen", "connect"] {
        set(
            &mut zconfig,
            &format!("transport/link/tls/{role}_certificate_base64"),
            &format!("\"{certificate}\""),
        )?;
        set_secret(
            &mut zconfig,
            &format!("transport/link/tls/{role}_private_key_base64"),
            quoted_private_key.as_str(),
        )?;
    }
    // Both sides present and verify a certificate chained to the authority.
    set(&mut zconfig, "transport/link/tls/enable_mtls", "true")?;
    // Members are reached by IP literal or through tunnels, and the member
    // name is enforced by access control instead.
    set(
        &mut zconfig,
        "transport/link/tls/verify_name_on_connect",
        "false",
    )?;
    // Certificates carry an unbounded validity window by design.
    set(
        &mut zconfig,
        "transport/link/tls/close_link_on_expiration",
        "false",
    )?;
    if membership == Membership::Enforced {
        set(
            &mut zconfig,
            "access_control",
            &access_control_policy(trust.member_name()),
        )?;
    }
    let secrets = vec![
        Zeroizing::new(trust.member_private_key_pem().to_owned()),
        private_key,
    ];
    Ok((zconfig, secrets))
}

/// A live Zenoh peer session plus the publishers/subscribers declared on it.
///
/// `recv` pulls the next inbound `(topic, payload)` delivered to any declared
/// subscriber, demuxed by the caller (the Python data-plane consumer keys on the
/// `command_id` carried inside the payload, exactly as the gossipsub path does).
pub struct ZenohSession {
    session: Session,
    // Publishers are kept behind `Arc` so `publish` can clone the handle out
    // under the lock and release the guard BEFORE the `put().await` (#309): the
    // lock then only guards the map, never an in-flight network put, so
    // concurrent publishes to different per-command keys don't serialize and a
    // `Block`-stalled put can't hold the map lock.
    publishers: Mutex<HashMap<String, Arc<Publisher<'static>>>>,
    subscribers: Mutex<HashMap<String, Subscriber<()>>>,
    inbound_tx: mpsc::Sender<(String, Vec<u8>)>,
    inbound_rx: Mutex<mpsc::Receiver<(String, Vec<u8>)>>,
    /// Certificate name of a legitimate cluster member (not secret).
    member_name: String,
    /// Multicast scouting group and port, when scouting is on (not secret).
    scouting_address: Option<String>,
}

impl ZenohSession {
    /// Return actual bound session locators without changing connectivity.
    pub async fn listen_addresses(&self) -> Vec<String> {
        self.session
            .info()
            .locators()
            .await
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Open the data-plane session for this process's cluster.
    ///
    /// Discovery scope and mutual-TLS trust derive from the libp2p cluster key
    /// (`SKULK_LIBP2P_NAMESPACE` plus `NETWORK_VERSION`), so the session can
    /// only ever form links with members of the same cluster.
    ///
    /// # Errors
    /// Fails on a non-`tls/` endpoint, invalid configuration, or a Zenoh open
    /// failure. Error text never contains key material.
    pub async fn open(config: ZenohConfig) -> AnyResult<Self> {
        let trust = crate::swarm::data_plane_trust()?;
        Self::open_with_trust(config, &trust, Membership::Enforced).await
    }

    async fn open_with_trust(
        config: ZenohConfig,
        trust: &DataPlaneTrust,
        membership: Membership,
    ) -> AnyResult<Self> {
        let (zconfig, secrets) = session_config(&config, trust, membership)?;
        let session = zenoh::open(zconfig)
            .await
            .map_err(|e| -> AnyError { redact(format!("zenoh open: {e}"), &secrets).into() })?;
        drop(secrets);
        let scouting_address = config.multicast_scouting.then(|| trust.scouting_address());
        let (inbound_tx, inbound_rx) = mpsc::channel(INBOUND_BUFFER_CAPACITY);
        Ok(Self {
            session,
            publishers: Mutex::new(HashMap::new()),
            subscribers: Mutex::new(HashMap::new()),
            inbound_tx,
            inbound_rx: Mutex::new(inbound_rx),
            member_name: trust.member_name().to_owned(),
            scouting_address,
        })
    }

    /// The multicast scouting address (`224.0.0.224:<port>`) this session
    /// uses, or `None` when scouting is off. Operators open this UDP port in
    /// host firewalls; it is derived from the cluster key and is not secret.
    #[must_use]
    pub fn scouting_address(&self) -> Option<&str> {
        self.scouting_address.as_deref()
    }

    /// Publish `data` on `topic` (Reliable + Block + single fixed priority).
    ///
    /// The publisher for a topic is declared once and reused, preserving the
    /// single-publisher-per-key FIFO ordering the data plane depends on.
    pub async fn publish(&self, topic: &str, data: Vec<u8>) -> AnyResult<()> {
        // Fast path: an existing publisher is cloned out under a short-lived
        // lock, which is then released before the put (#309). The guard never
        // spans the `put().await`, so a Block-stalled put can't hold the map.
        let existing = self.publishers.lock().await.get(topic).cloned();
        let publisher = match existing {
            Some(p) => p,
            None => {
                // Declare WITHOUT holding the lock across the await, then insert
                // with a double-check: a concurrent publish to the same key may
                // have declared first, in which case we keep the stored one (and
                // drop ours) so there stays exactly one publisher per key per
                // session (the single-publisher FIFO ordering the plane needs).
                let declared = Arc::new(
                    self.session
                        .declare_publisher(topic.to_string())
                        .congestion_control(CongestionControl::Block)
                        .priority(Priority::Data)
                        .reliability(Reliability::Reliable)
                        .await
                        .map_err(|e| -> AnyError {
                            format!("declare_publisher {topic}: {e}").into()
                        })?,
                );
                let mut publishers = self.publishers.lock().await;
                publishers
                    .entry(topic.to_string())
                    .or_insert(declared)
                    .clone()
            }
        };
        publisher
            .put(data)
            .await
            .map_err(|e| -> AnyError { format!("publish {topic}: {e}").into() })
    }

    /// Declare a subscriber on `topic`; inbound samples are forwarded to `recv`.
    ///
    /// Idempotent: subscribing to an already-subscribed topic is a no-op.
    pub async fn subscribe(&self, topic: &str) -> AnyResult<()> {
        // Check-and-release before declaring so the lock never spans the await
        // (#309); startup-only, but tidy alongside the publish refactor.
        if self.subscribers.lock().await.contains_key(topic) {
            return Ok(());
        }
        let tx = self.inbound_tx.clone();
        let subscriber = self
            .session
            .declare_subscriber(topic.to_string())
            .callback(move |sample| {
                let key = sample.key_expr().as_str().to_string();
                let payload = sample.payload().to_bytes().to_vec();
                // The Zenoh callback runs on a Zenoh thread and must not block,
                // so use the non-blocking try_send. On a full channel (consumer
                // fell behind) drop the sample rather than grow memory without
                // bound; on a closed channel (session teardown) stay silent.
                match tx.try_send((key, payload)) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        log::warn!("zenoh data-plane inbound buffer full; dropping a chunk");
                    }
                    Err(TrySendError::Closed(_)) => {}
                }
            })
            .await
            .map_err(|e| -> AnyError { format!("declare_subscriber {topic}: {e}").into() })?;
        // Re-lock and insert with a double-check: if a concurrent subscribe to
        // the same topic won the race, keep the existing one and drop ours.
        let mut subscribers = self.subscribers.lock().await;
        subscribers.entry(topic.to_string()).or_insert(subscriber);
        Ok(())
    }

    /// Await the next inbound `(topic, payload)`, or `None` once the session is
    /// closed and all senders are dropped.
    pub async fn recv(&self) -> Option<(String, Vec<u8>)> {
        let mut rx = self.inbound_rx.lock().await;
        rx.recv().await
    }

    /// Count the cluster members this session currently holds a live,
    /// authenticated transport to.
    ///
    /// This is the data plane's only connectivity ground truth: a node whose
    /// count stays at zero while cluster peers advertise Zenoh is isolated
    /// (e.g. a zero-config remote member that multicast scouting cannot
    /// reach), and every remote stream to or from it dies with transport
    /// errors while the control plane still looks healthy. Surfacing the
    /// count lets Python advertise isolation instead of failing silently.
    ///
    /// Only peers whose every link presents the cluster's member certificate
    /// count. A transport that access control quarantines carries no traffic,
    /// so counting it would mask exactly the isolation this number reports.
    pub async fn connected_peer_count(&self) -> usize {
        let mut members: HashMap<ZenohId, bool> = HashMap::new();
        for link in self.session.info().links().await {
            let authenticated = link.auth_identifier() == Some(self.member_name.as_str());
            members
                .entry(*link.zid())
                .and_modify(|all| *all = *all && authenticated)
                .or_insert(authenticated);
        }
        members.into_values().filter(|all| *all).count()
    }
}

#[cfg(test)]
mod tests {
    //! Session-level tests over loopback with explicit endpoints, so they need
    //! no multicast and run anywhere. The multicast discovery test is ignored
    //! by default because it depends on the host's multicast routing; run it
    //! with `cargo test -p networking -- --ignored`.

    use std::future::Future;
    use std::time::Duration;

    use tokio::time::{Instant, sleep, timeout};

    use super::{Membership, ZenohConfig, ZenohSession};
    use crate::data_plane_trust::DataPlaneTrust;
    use crate::swarm::preshared_key_for;

    const NAMESPACE: &str = "nsdataplanetest";
    const SETTLE: Duration = Duration::from_secs(3);
    const DEADLINE: Duration = Duration::from_secs(15);

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .map(|address| address.port())
            .expect("free port")
    }

    fn listener(port: u16) -> ZenohConfig {
        ZenohConfig {
            listen_endpoints: vec![format!("tls/127.0.0.1:{port}")],
            connect_endpoints: vec![],
            multicast_scouting: false,
            namespace: Some(NAMESPACE.to_owned()),
        }
    }

    fn dialer(ports: &[u16]) -> ZenohConfig {
        ZenohConfig {
            listen_endpoints: vec![],
            connect_endpoints: ports
                .iter()
                .map(|port| format!("tls/127.0.0.1:{port}"))
                .collect(),
            multicast_scouting: false,
            namespace: Some(NAMESPACE.to_owned()),
        }
    }

    fn trust(namespace: &str) -> DataPlaneTrust {
        DataPlaneTrust::derive(&preshared_key_for(Some(namespace))).expect("derive trust")
    }

    async fn member(config: ZenohConfig, trust: &DataPlaneTrust) -> ZenohSession {
        ZenohSession::open_with_trust(config, trust, Membership::Enforced)
            .await
            .expect("open member session")
    }

    /// Every transport, authenticated or not.
    async fn raw_transports(session: &ZenohSession) -> usize {
        session.session.info().peers_zid().await.count()
    }

    async fn eventually<F, Fut>(mut check: F) -> bool
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = bool>,
    {
        let deadline = Instant::now() + DEADLINE;
        loop {
            if check().await {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    /// Publish repeatedly until `subscriber` receives a sample or `within`
    /// passes. Subscriber declarations propagate a moment after a transport
    /// forms, so a single put would make the positive cases flaky.
    async fn deliver(
        publisher: &ZenohSession,
        subscriber: &ZenohSession,
        key: &str,
        payload: &[u8],
        within: Duration,
    ) -> Option<(String, Vec<u8>)> {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            publisher
                .publish(key, payload.to_vec())
                .await
                .expect("publish");
            if let Ok(Some(sample)) = timeout(Duration::from_millis(200), subscriber.recv()).await {
                return Some(sample);
            }
        }
        None
    }

    /// Drain `subscriber` for `within` and report whether `payload` arrived.
    async fn received(subscriber: &ZenohSession, payload: &[u8], within: Duration) -> bool {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if let Ok(Some((_, data))) =
                timeout(Duration::from_millis(200), subscriber.recv()).await
            {
                if data == payload {
                    return true;
                }
            }
        }
        false
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn same_key_members_connect_and_exchange_samples() {
        let port = free_port();
        let accepting = member(listener(port), &trust("cluster-a")).await;
        let dialing = member(dialer(&[port]), &trust("cluster-a")).await;
        accepting
            .subscribe("data/accepting")
            .await
            .expect("subscribe");

        let sample = deliver(&dialing, &accepting, "data/accepting", b"hello", DEADLINE)
            .await
            .expect("a same-key member delivers samples");
        // The namespace prefix is applied and stripped transparently.
        assert_eq!(sample, ("data/accepting".to_owned(), b"hello".to_vec()));
        assert!(eventually(|| async { accepting.connected_peer_count().await == 1 }).await);
        assert_eq!(dialing.connected_peer_count().await, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn different_keys_fail_the_handshake_and_deliver_nothing() {
        let port = free_port();
        let accepting = member(listener(port), &trust("cluster-a")).await;
        let foreign = member(dialer(&[port]), &trust("cluster-b")).await;
        accepting
            .subscribe("data/accepting")
            .await
            .expect("subscribe");

        let leaked = deliver(&foreign, &accepting, "data/accepting", b"cross", SETTLE).await;
        assert!(
            leaked.is_none(),
            "a different cluster key delivered a sample"
        );
        assert_eq!(raw_transports(&accepting).await, 0);
        assert_eq!(raw_transports(&foreign).await, 0);
        assert_eq!(accepting.connected_peer_count().await, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plaintext_and_certificateless_clients_cannot_connect() {
        let port = free_port();
        let accepting = member(listener(port), &trust("cluster-a")).await;

        // A stock Zenoh peer on plaintext TCP, and a TLS client that trusts the
        // authority but holds no member certificate.
        let mut plain = zenoh::Config::default();
        let mut certificateless = zenoh::Config::default();
        for (config, endpoint) in [
            (&mut plain, format!("tcp/127.0.0.1:{port}")),
            (&mut certificateless, format!("tls/127.0.0.1:{port}")),
        ] {
            config.insert_json5("mode", "\"peer\"").expect("mode");
            config
                .insert_json5("scouting/multicast/enabled", "false")
                .expect("scouting");
            config
                .insert_json5("listen/endpoints", "[]")
                .expect("listen");
            config
                .insert_json5("connect/endpoints", &format!("[{endpoint:?}]"))
                .expect("connect");
        }
        let authority = {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD
                .encode(trust("cluster-a").authority_certificate_pem().as_bytes())
        };
        certificateless
            .insert_json5(
                "transport/link/tls/root_ca_certificate_base64",
                &format!("{authority:?}"),
            )
            .expect("root");
        certificateless
            .insert_json5("transport/link/tls/verify_name_on_connect", "false")
            .expect("verify");

        let mut outsiders = Vec::new();
        for config in [plain, certificateless] {
            // A peer whose connect fails may still open; either way it must not
            // hold a transport to the member.
            if let Ok(session) = zenoh::open(config).await {
                outsiders.push(session);
            }
        }
        for outsider in &outsiders {
            let _subscriber = outsider
                .declare_subscriber("**")
                .await
                .expect("outsider subscriber");
        }
        sleep(SETTLE).await;
        accepting
            .publish("data/anyone", b"private".to_vec())
            .await
            .expect("publish");
        assert_eq!(raw_transports(&accepting).await, 0);
        for outsider in &outsiders {
            assert_eq!(outsider.info().peers_zid().await.count(), 0);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn peer_without_the_member_name_is_quarantined() {
        // Zenoh's dialer also trusts the public Web PKI roots, so an acceptor
        // with any publicly trusted certificate passes its chain check. Model
        // that with a certificate chained to the real authority but carrying a
        // foreign name, on a peer that applies no policy of its own: the TLS
        // handshake succeeds, and only access control stands in the way.
        let key = preshared_key_for(Some("cluster-a"));
        let intruder_port = free_port();
        let intruder_trust = DataPlaneTrust::derive_with_certificate_name(&key, "intruder.example")
            .expect("intruder trust");
        let intruder_config = ZenohConfig {
            namespace: None,
            ..listener(intruder_port)
        };
        let intruder =
            ZenohSession::open_with_trust(intruder_config, &intruder_trust, Membership::Unenforced)
                .await
                .expect("intruder session");
        intruder.subscribe("**").await.expect("intruder subscribe");

        let peer_port = free_port();
        let peer = member(listener(peer_port), &trust("cluster-a")).await;
        peer.subscribe("data/peer").await.expect("subscribe");
        let victim = member(dialer(&[intruder_port, peer_port]), &trust("cluster-a")).await;
        victim.subscribe("data/victim").await.expect("subscribe");

        // The honest pair works: the victim's output reaches its real peer.
        let sample = deliver(&victim, &peer, "data/peer", b"secret", DEADLINE).await;
        assert_eq!(sample, Some(("data/peer".to_owned(), b"secret".to_vec())));
        // The intruder holds a transport, yet it is not counted as a member.
        assert!(eventually(|| async { raw_transports(&victim).await == 2 }).await);
        assert_eq!(victim.connected_peer_count().await, 1);
        // The victim never routed its output to the intruder's wildcard
        // subscriber.
        assert!(!received(&intruder, b"secret", Duration::from_millis(500)).await);
        // And the intruder cannot inject into the victim's stream.
        let forged_key = format!("{NAMESPACE}/data/victim");
        for _ in 0..10 {
            intruder
                .publish(&forged_key, b"forged".to_vec())
                .await
                .expect("intruder publish");
            sleep(Duration::from_millis(100)).await;
        }
        assert!(!received(&victim, b"forged", SETTLE).await);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_tls_endpoints_are_refused_before_opening() {
        let error = ZenohSession::open_with_trust(
            ZenohConfig {
                listen_endpoints: vec!["tcp/127.0.0.1:0".to_owned()],
                ..ZenohConfig::default()
            },
            &trust("cluster-a"),
            Membership::Enforced,
        )
        .await
        .err()
        .expect("plaintext listener refused");
        assert!(error.to_string().contains("tls/"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "needs working local multicast; run explicitly with --ignored"]
    async fn multicast_discovery_is_scoped_to_the_scouting_port() {
        fn scouting() -> ZenohConfig {
            ZenohConfig {
                listen_endpoints: vec!["tls/127.0.0.1:0".to_owned()],
                connect_endpoints: vec![],
                multicast_scouting: true,
                namespace: Some(NAMESPACE.to_owned()),
            }
        }
        // A unique namespace keeps this run off any real cluster's port.
        let namespace = format!("scout-{}", std::process::id());
        let own = trust(&namespace);
        assert!(own.scouting_address().starts_with("224.0.0.224:"));
        let first = member(scouting(), &own).await;
        let second = member(scouting(), &trust(&namespace)).await;
        assert_eq!(
            first.scouting_address(),
            Some(own.scouting_address().as_str())
        );
        assert!(
            eventually(|| async { first.connected_peer_count().await == 1 }).await,
            "same-key peers on the derived port discover each other"
        );

        // Same key, so TLS would succeed, but another scouting port: the
        // stranger never discovers or dials either member. This isolates the
        // discovery boundary from the handshake boundary.
        let derived = preshared_key_for(Some(&namespace));
        let other_port = super::super::data_plane_trust::scouting_port(&derived).wrapping_add(1);
        let stranger_trust = DataPlaneTrust::derive(&derived)
            .expect("derive")
            .with_scouting_port(other_port);
        let stranger = member(scouting(), &stranger_trust).await;
        sleep(SETTLE).await;
        assert_eq!(raw_transports(&stranger).await, 0);
        assert_eq!(raw_transports(&first).await, 1);
        assert_eq!(raw_transports(&second).await, 1);
    }
}
