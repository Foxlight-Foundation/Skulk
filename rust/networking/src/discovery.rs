use crate::ext::MultiaddrExt;
use delegate::delegate;
use either::Either;
use futures_lite::FutureExt;
use futures_timer::Delay;
use libp2p::core::transport::PortUse;
use libp2p::core::{ConnectedPoint, Endpoint};
use libp2p::swarm::behaviour::ConnectionEstablished;
use libp2p::swarm::dial_opts::DialOpts;
use libp2p::swarm::{
    CloseConnection, ConnectionClosed, ConnectionDenied, ConnectionHandler,
    ConnectionHandlerSelect, ConnectionId, FromSwarm, NetworkBehaviour, THandler, THandlerInEvent,
    THandlerOutEvent, ToSwarm, dummy,
};
use libp2p::{Multiaddr, PeerId, identity, mdns};
use std::collections::{BTreeSet, HashMap};
use std::convert::Infallible;
use std::io;
use std::net::IpAddr;
use std::task::{Context, Poll};
use std::time::Duration;
use util::wakerdeque::WakerDeque;

const RETRY_CONNECT_INTERVAL: Duration = Duration::from_secs(5);
const CONNECTED_PEER_LINK_LOCAL_RETRY_ROUNDS: u64 = 12;
const MAX_CONSECUTIVE_PING_FAILURES: u8 = 3;

mod managed {
    use libp2p::swarm::NetworkBehaviour;
    use libp2p::{identity, mdns, ping};
    use std::io;
    use std::time::Duration;

    const MDNS_RECORD_TTL: Duration = Duration::from_secs(2_500);
    const MDNS_QUERY_INTERVAL: Duration = Duration::from_secs(1_500);
    const PING_TIMEOUT: Duration = Duration::from_secs(5);
    const PING_INTERVAL: Duration = Duration::from_secs(5);

    #[derive(NetworkBehaviour)]
    pub struct Behaviour {
        mdns: mdns::tokio::Behaviour,
        ping: ping::Behaviour,
    }

    impl Behaviour {
        pub fn new(keypair: &identity::Keypair) -> io::Result<Self> {
            Ok(Self {
                mdns: mdns_behaviour(keypair)?,
                ping: ping_behaviour(),
            })
        }
    }

    fn mdns_behaviour(keypair: &identity::Keypair) -> io::Result<mdns::tokio::Behaviour> {
        use mdns::{Config, tokio};

        // mDNS config => enable IPv6
        let mdns_config = Config {
            ttl: MDNS_RECORD_TTL,
            query_interval: MDNS_QUERY_INTERVAL,

            // enable_ipv6: true, // TODO: for some reason, TCP+mDNS don't work well with ipv6?? figure out how to make work
            ..Default::default()
        };

        let mdns_behaviour = tokio::Behaviour::new(mdns_config, keypair.public().to_peer_id());
        Ok(mdns_behaviour?)
    }

    fn ping_behaviour() -> ping::Behaviour {
        ping::Behaviour::new(
            ping::Config::new()
                .with_timeout(PING_TIMEOUT)
                .with_interval(PING_INTERVAL),
        )
    }
}

/// Events for when a listening connection is truly established and truly closed.
#[derive(Debug, Clone)]
pub enum Event {
    ConnectionEstablished {
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    },
    ConnectionClosed {
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    },
}

/// Discovery behavior that wraps mDNS to produce truly discovered durable peer-connections.
///
/// The behaviour operates as such:
///  1) All true (listening) connections/disconnections are tracked, emitting corresponding events
///     to the swarm.
///  1) mDNS discovered/expired peers are tracked; discovered but not connected peers are dialed
///     immediately, and expired but connected peers are disconnected from immediately.
///  2) Every fixed interval: discovered but not connected peers are dialed, and expired but
///     connected peers are disconnected from.
pub struct Behaviour {
    // state-tracking for managed behaviors & mDNS-discovered peers
    managed: managed::Behaviour,
    mdns_discovered: HashMap<PeerId, BTreeSet<Multiaddr>>,
    bootstrap_peers: Vec<Multiaddr>,

    // Remote IPs we currently hold at least one connection to, ref-counted by
    // connection. Bootstrap peers are dialed by address (no peer id, so libp2p
    // cannot dedup them); without this, the periodic retry opens a *new*
    // connection to each bootstrap peer every interval and never stops, leaking
    // sockets unboundedly. We skip re-dialing a bootstrap address whose IP is
    // already connected, and resume once it drops.
    connected_ips: HashMap<IpAddr, usize>,

    // A peer can be connected on several interfaces. Once any path is live,
    // failed link-local addresses are retried at a slower cadence instead of
    // every five seconds. The initial mDNS discovery still dials every address,
    // preserving direct Thunderbolt paths when they are actually routable.
    connected_peers: HashMap<PeerId, usize>,

    // One transiently delayed ping must not tear down a healthy connection.
    // Failure counts are per socket because a multi-homed peer can have one bad
    // path while its other paths remain healthy.
    ping_failures: HashMap<(PeerId, ConnectionId), u8>,

    retry_round: u64,

    retry_delay: Delay, // retry interval

    // pending events to emmit => waker-backed Deque to control polling
    pending_events: WakerDeque<ToSwarm<Event, Infallible>>,
}

impl Behaviour {
    pub fn new(keypair: &identity::Keypair, bootstrap_peers: Vec<Multiaddr>) -> io::Result<Self> {
        Ok(Self {
            managed: managed::Behaviour::new(keypair)?,
            mdns_discovered: HashMap::new(),
            bootstrap_peers,
            connected_ips: HashMap::new(),
            connected_peers: HashMap::new(),
            ping_failures: HashMap::new(),
            retry_round: 0,
            retry_delay: Delay::new(RETRY_CONNECT_INTERVAL),
            pending_events: WakerDeque::new(),
        })
    }

    fn dial(&mut self, peer_id: PeerId, addr: Multiaddr) {
        self.pending_events.push_back(ToSwarm::Dial {
            opts: DialOpts::peer_id(peer_id).addresses(vec![addr]).build(),
        })
    }

    fn close_connection(&mut self, peer_id: PeerId, connection: ConnectionId) {
        // push front to make this IMMEDIATE
        self.pending_events.push_front(ToSwarm::CloseConnection {
            peer_id,
            connection: CloseConnection::One(connection),
        })
    }

    fn handle_mdns_discovered(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (p, ma) in peers {
            self.dial(p, ma.clone()); // always connect

            // get peer's multi-addresses or insert if missing
            let Some(mas) = self.mdns_discovered.get_mut(&p) else {
                self.mdns_discovered.insert(p, BTreeSet::from([ma]));
                continue;
            };

            // multiaddress should never already be present - else something has gone wrong
            let is_new_addr = mas.insert(ma);
            assert!(is_new_addr, "cannot discover a discovered peer");
        }
    }

    fn handle_mdns_expired(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (p, ma) in peers {
            // at this point, we *must* have the peer
            let mas = self
                .mdns_discovered
                .get_mut(&p)
                .expect("nonexistent peer cannot expire");

            // at this point, we *must* have the multiaddress
            let was_present = mas.remove(&ma);
            assert!(was_present, "nonexistent multiaddress cannot expire");

            // if empty, remove the peer-id entirely
            if mas.is_empty() {
                self.mdns_discovered.remove(&p);
            }
        }
    }

    fn on_connection_established(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        // track the connected IP so the retry loop stops re-dialing a bootstrap
        // peer we already reached (see `connected_ips`)
        *self.connected_ips.entry(remote_ip).or_insert(0) += 1;
        *self.connected_peers.entry(peer_id).or_insert(0) += 1;

        // send out connected event
        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionEstablished {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
    }

    fn on_connection_closed(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        // drop the IP from the connected set once its last connection closes, so
        // the retry loop will dial that bootstrap peer again
        if let Some(count) = self.connected_ips.get_mut(&remote_ip) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.connected_ips.remove(&remote_ip);
            }
        }
        if let Some(count) = self.connected_peers.get_mut(&peer_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.connected_peers.remove(&peer_id);
            }
        }
        self.ping_failures.remove(&(peer_id, connection_id));

        // send out disconnected event
        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionClosed {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
    }

    fn record_ping_result(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        succeeded: bool,
    ) -> bool {
        let key = (peer_id, connection_id);
        if succeeded {
            self.ping_failures.remove(&key);
            return false;
        }

        let failures = self.ping_failures.entry(key).or_insert(0);
        *failures = failures.saturating_add(1);
        if *failures < MAX_CONSECUTIVE_PING_FAILURES {
            return false;
        }
        self.ping_failures.remove(&key);
        true
    }

    fn should_retry_mdns_address(&self, peer_id: &PeerId, addr: &Multiaddr) -> bool {
        let is_link_local = addr.try_to_tcp_addr().is_some_and(|(ip, _)| match ip {
            IpAddr::V4(ip) => ip.is_link_local(),
            IpAddr::V6(ip) => ip.is_unicast_link_local(),
        });
        if !is_link_local || !self.connected_peers.contains_key(peer_id) {
            return true;
        }

        self.retry_round
            .is_multiple_of(CONNECTED_PEER_LINK_LOCAL_RETRY_ROUNDS)
    }
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler =
        ConnectionHandlerSelect<dummy::ConnectionHandler, THandler<managed::Behaviour>>;
    type ToSwarm = Event;

    // simply delegate to underlying mDNS behaviour

    delegate! {
        to self.managed {
            fn handle_pending_inbound_connection(&mut self, connection_id: ConnectionId, local_addr: &Multiaddr, remote_addr: &Multiaddr) -> Result<(), ConnectionDenied>;
            fn handle_pending_outbound_connection(&mut self, connection_id: ConnectionId, maybe_peer: Option<PeerId>, addresses: &[Multiaddr], effective_role: Endpoint) -> Result<Vec<Multiaddr>, ConnectionDenied>;
        }
    }

    fn handle_established_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(ConnectionHandler::select(
            dummy::ConnectionHandler,
            self.managed.handle_established_inbound_connection(
                connection_id,
                peer,
                local_addr,
                remote_addr,
            )?,
        ))
    }

    #[allow(clippy::needless_question_mark)]
    fn handle_established_outbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        addr: &Multiaddr,
        role_override: Endpoint,
        port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(ConnectionHandler::select(
            dummy::ConnectionHandler,
            self.managed.handle_established_outbound_connection(
                connection_id,
                peer,
                addr,
                role_override,
                port_use,
            )?,
        ))
    }

    fn on_connection_handler_event(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        event: THandlerOutEvent<Self>,
    ) {
        match event {
            Either::Left(ev) => libp2p::core::util::unreachable(ev),
            Either::Right(ev) => {
                self.managed
                    .on_connection_handler_event(peer_id, connection_id, ev)
            }
        }
    }

    // hook into these methods to drive behavior

    fn on_swarm_event(&mut self, event: FromSwarm) {
        self.managed.on_swarm_event(event); // let mDNS handle swarm events

        // handle swarm events to update internal state:
        match event {
            FromSwarm::ConnectionEstablished(ConnectionEstablished {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                let remote_address = match endpoint {
                    ConnectedPoint::Dialer { address, .. } => address,
                    ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                };

                if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                    // handle connection established event which is filtered correctly
                    self.on_connection_established(peer_id, connection_id, ip, port)
                }
            }
            FromSwarm::ConnectionClosed(ConnectionClosed {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                let remote_address = match endpoint {
                    ConnectedPoint::Dialer { address, .. } => address,
                    ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                };

                if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                    // handle connection closed event which is filtered correctly
                    self.on_connection_closed(peer_id, connection_id, ip, port)
                }
            }

            // since we are running TCP/IP transport layer, we are assuming that
            // no address changes can occur, hence encountering one is a fatal error
            FromSwarm::AddressChange(a) => {
                unreachable!("unhandlable: address change encountered: {:?}", a)
            }
            _ => {}
        }
    }

    fn poll(&mut self, cx: &mut Context) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        // delegate to managed behaviors for any behaviors they need to perform
        match self.managed.poll(cx) {
            Poll::Ready(ToSwarm::GenerateEvent(e)) => {
                match e {
                    // handle discovered and expired events from mDNS
                    managed::BehaviourEvent::Mdns(e) => match e.clone() {
                        mdns::Event::Discovered(peers) => {
                            self.handle_mdns_discovered(peers);
                        }
                        mdns::Event::Expired(peers) => {
                            self.handle_mdns_expired(peers);
                        }
                    },

                    // handle ping events => if error then disconnect
                    managed::BehaviourEvent::Ping(e) => {
                        if self.record_ping_result(e.peer, e.connection, e.result.is_ok()) {
                            self.close_connection(e.peer, e.connection)
                        }
                    }
                }

                // since we just consumed an event, we should immediately wake just in case
                // there are more events to come where that came from
                cx.waker().wake_by_ref();
            }

            // forward any other mDNS event to the swarm or its connection handler(s)
            Poll::Ready(e) => {
                return Poll::Ready(
                    e.map_out(|_| unreachable!("events returning to swarm already handled"))
                        .map_in(Either::Right),
                );
            }

            Poll::Pending => {}
        }

        // retry connecting to all mDNS peers periodically (fails safely if already connected)
        if self.retry_delay.poll(cx).is_ready() {
            self.retry_round = self.retry_round.wrapping_add(1);
            for (p, mas) in self.mdns_discovered.clone() {
                for ma in mas {
                    if self.should_retry_mdns_address(&p, &ma) {
                        self.dial(p, ma)
                    }
                }
            }
            // dial bootstrap peers (for environments where mDNS is unavailable),
            // skipping any whose IP we already hold a connection to — bootstrap
            // dials carry no peer id and so cannot be deduped by libp2p, so this
            // guard is what prevents the retry loop from leaking a new socket to
            // each bootstrap peer every interval.
            let bootstrap_to_dial: Vec<Multiaddr> = self
                .bootstrap_peers
                .iter()
                .filter(|addr| match addr.try_to_tcp_addr() {
                    Some((ip, _)) => !self.connected_ips.contains_key(&ip),
                    None => true,
                })
                .cloned()
                .collect();
            for addr in bootstrap_to_dial {
                self.pending_events.push_back(ToSwarm::Dial {
                    opts: DialOpts::unknown_peer_id().address(addr).build(),
                })
            }
            self.retry_delay.reset(RETRY_CONNECT_INTERVAL) // reset timeout
        }

        // send out any pending events from our own service
        if let Some(e) = self.pending_events.pop_front(cx) {
            return Poll::Ready(e.map_in(Either::Left));
        }

        // wait for pending events
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn behaviour() -> Behaviour {
        let keypair = identity::Keypair::generate_ed25519();
        Behaviour::new(&keypair, vec![]).expect("discovery behaviour")
    }

    #[test]
    fn ping_requires_three_consecutive_failures_before_close() {
        let mut behaviour = behaviour();
        let peer = identity::Keypair::generate_ed25519().public().to_peer_id();
        let connection = ConnectionId::new_unchecked(1);

        assert!(!behaviour.record_ping_result(peer, connection, false));
        assert!(!behaviour.record_ping_result(peer, connection, false));
        assert!(!behaviour.record_ping_result(peer, connection, true));
        assert!(!behaviour.record_ping_result(peer, connection, false));
        assert!(!behaviour.record_ping_result(peer, connection, false));
        assert!(behaviour.record_ping_result(peer, connection, false));
    }

    #[test]
    fn connected_peer_link_local_retries_are_slow_but_not_disabled() {
        let mut behaviour = behaviour();
        let peer = identity::Keypair::generate_ed25519().public().to_peer_id();
        let link_local: Multiaddr = "/ip4/169.254.10.2/tcp/52416"
            .parse()
            .expect("link-local multiaddr");
        let routable: Multiaddr = "/ip4/192.168.1.2/tcp/52416"
            .parse()
            .expect("routable multiaddr");

        assert!(behaviour.should_retry_mdns_address(&peer, &link_local));
        behaviour.connected_peers.insert(peer, 1);
        behaviour.retry_round = 1;
        assert!(!behaviour.should_retry_mdns_address(&peer, &link_local));
        assert!(behaviour.should_retry_mdns_address(&peer, &routable));

        behaviour.retry_round = CONNECTED_PEER_LINK_LOCAL_RETRY_ROUNDS;
        assert!(behaviour.should_retry_mdns_address(&peer, &link_local));
    }
}
