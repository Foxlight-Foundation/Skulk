//! Cluster-scoped discovery and mutual-TLS trust for the Zenoh data plane.
//!
//! The libp2p control plane is gated by a private-network pre-shared key
//! (`swarm.rs`, `PNET_PRESHARED_KEY`). The Zenoh data plane carries the same
//! cluster's generated output and media, so it is gated by the same secret.
//! Everything here derives one-way from that 32-byte cluster key, so the two
//! planes stay keyed together by construction and a `NETWORK_VERSION` or
//! namespace change re-keys both:
//!
//! - **Discovery scope.** The multicast scouting port is a domain-separated
//!   hash of the key, so clusters with different keys normally scout on
//!   different ports of the shared `224.0.0.224` group. It is a hash into about
//!   7,000 ports, so two keys occasionally share one; the authority below, not
//!   the port, is what keeps a foreign cluster out.
//! - **Authority.** Every node independently derives the same Ed25519
//!   certificate authority. Each process then issues itself an ephemeral member
//!   certificate signed by it. Zenoh's TLS link verifies the peer chain in both
//!   directions (`enable_mtls`), so a device without the key cannot complete a
//!   handshake, and a passive observer sees only TLS 1.3.
//! - **Member name.** Zenoh's TLS dialer merges the public Web PKI roots into
//!   its trust store, so a chain check alone would accept an acceptor holding
//!   any publicly trusted certificate. Member certificates therefore carry a
//!   cluster-specific common name under the reserved `.invalid` top-level
//!   domain, which no public authority can issue, and the session's access
//!   control admits traffic only from links whose peer presents that name.
//!
//! Secret material (the derived seed and the authority's private key) never
//! leaves this module: Zenoh receives only the public authority certificate and
//! the process's ephemeral member key and certificate.

use std::fmt;

use hkdf::Hkdf;
use rcgen::{
    BasicConstraints, CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa,
    Issuer, KeyPair, KeyUsagePurpose, SanType, date_time_ymd,
};
use sha2::{Digest as _, Sha256};
use zeroize::Zeroizing;

use crate::alias::{AnyError, AnyResult};

/// HKDF `info` for the authority seed. Changing it re-keys every cluster's data
/// plane, so it moves only with a `NETWORK_VERSION` bump.
const AUTHORITY_SEED_INFO: &[u8] = b"skulk-zenoh-ca-v1";

/// Domain separator for the scouting-port hash, so the port can never be
/// confused with (or used to recompute) any other value derived from the key.
const SCOUTING_PORT_DOMAIN: &[u8] = b"skulk-zenoh-scout-v1\0";

/// The shared multicast group. 224.0.0.0/24 is link-local control space, which
/// IGMP-snooping switches always flood, so discovery works without a querier.
const SCOUTING_GROUP: &str = "224.0.0.224";

/// Scouting ports come from `[SCOUTING_PORT_BASE, SCOUTING_PORT_BASE + SPAN)`.
/// The range sits below every common ephemeral range (Linux 32768+, macOS and
/// Windows 49152+) so an outgoing socket cannot occupy it by chance, clears
/// Zenoh's own 7446/7447 defaults and Steam's 27000 block, and the skip list
/// below steps over the well-known UDP listeners inside it. A bind conflict on
/// Linux would fail session open, so these choices matter.
const SCOUTING_PORT_BASE: u16 = 20_000;
const SCOUTING_PORT_SPAN: u16 = 7_000;
/// Syncthing local discovery (21027) and transfer (22000).
const SCOUTING_PORT_SKIP: [u16; 2] = [21_027, 22_000];

const AUTHORITY_COMMON_NAME: &str = "Skulk data-plane authority";
const ORGANIZATION_NAME: &str = "Skulk";
const MEMBER_NAME_PREFIX: &str = "skulk-dataplane-";
const MEMBER_NAME_SUFFIX: &str = ".invalid";
/// Bytes of the authority public-key digest carried in the member name.
const MEMBER_NAME_DIGEST_BYTES: usize = 16;

/// RFC 8410 PKCS#8 v1 header for a raw 32-byte Ed25519 private key.
const ED25519_PKCS8_V1_PREFIX: [u8; 16] = [
    0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20,
];

/// Derive the multicast scouting port for a cluster key.
///
/// Deterministic, so every node of a cluster lands on the same port. Two
/// clusters collide with probability about 1/7000, in which case they fall
/// back to meeting at the scouting layer and failing the TLS handshake, which
/// is still safe. The port is visible on the wire and reveals only a 13-bit
/// domain-separated hash of the key.
#[must_use]
pub fn scouting_port(cluster_key: &[u8; 32]) -> u16 {
    let digest = Sha256::new()
        .chain_update(SCOUTING_PORT_DOMAIN)
        .chain_update(cluster_key)
        .finalize();
    let mut prefix = [0_u8; 4];
    prefix.copy_from_slice(digest.get(..4).unwrap_or(&[0; 4]));
    let base = u32::from(SCOUTING_PORT_BASE);
    let span = u32::from(SCOUTING_PORT_SPAN);
    // `span` is a nonzero constant and `base + span` fits in u16, so neither
    // the remainder nor the additions below can fail or overflow.
    let mut offset = u32::from_be_bytes(prefix).checked_rem(span).unwrap_or(0);
    // Step past a well-known listener; the skip list is tiny, so this loop
    // runs at most a couple of times.
    while SCOUTING_PORT_SKIP
        .iter()
        .any(|skip| u32::from(*skip) == base.saturating_add(offset))
    {
        offset = offset.saturating_add(1).checked_rem(span).unwrap_or(0);
    }
    u16::try_from(base.saturating_add(offset)).unwrap_or(SCOUTING_PORT_BASE)
}

/// Derive the authority seed: HKDF-SHA256 over the cluster key.
fn derive_authority_seed(cluster_key: &[u8; 32]) -> AnyResult<Zeroizing<[u8; 32]>> {
    let mut seed = Zeroizing::new([0_u8; 32]);
    Hkdf::<Sha256>::new(None, cluster_key)
        .expand(AUTHORITY_SEED_INFO, seed.as_mut())
        .map_err(|_| -> AnyError { "data-plane authority seed derivation failed".into() })?;
    Ok(seed)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Wide validity window so clock skew (or a node booting without a real-time
/// clock) never fails a handshake. Trust is rooted in the key, not in expiry:
/// rotating the cluster namespace re-keys the authority.
fn set_unbounded_validity(params: &mut CertificateParams) {
    params.not_before = date_time_ymd(1970, 1, 1);
    params.not_after = date_time_ymd(9999, 12, 31);
}

/// The deterministic cluster authority. Holds the private key, so it is never
/// stored beyond issuing this process's member certificate.
struct ClusterAuthority {
    issuer: Issuer<'static, KeyPair>,
    certificate_pem: String,
    member_name: String,
}

impl ClusterAuthority {
    fn derive(cluster_key: &[u8; 32]) -> AnyResult<Self> {
        let seed = derive_authority_seed(cluster_key)?;
        let mut pkcs8 = Zeroizing::new(Vec::with_capacity(48));
        pkcs8.extend_from_slice(&ED25519_PKCS8_V1_PREFIX);
        pkcs8.extend_from_slice(seed.as_ref());
        // The error is deliberately generic: nothing about the key may reach
        // an error string that Python could log.
        let key = KeyPair::try_from(pkcs8.as_slice())
            .map_err(|_| -> AnyError { "data-plane authority key derivation failed".into() })?;

        let mut name = DistinguishedName::new();
        name.push(DnType::CommonName, AUTHORITY_COMMON_NAME);
        name.push(DnType::OrganizationName, ORGANIZATION_NAME);
        let mut params = CertificateParams::default();
        params.distinguished_name = name;
        params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
        params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::CrlSign,
            KeyUsagePurpose::DigitalSignature,
        ];
        set_unbounded_validity(&mut params);
        // Ed25519 signatures and rcgen's key-derived serial number are both
        // deterministic, so every node produces a byte-identical certificate.
        let certificate = params
            .self_signed(&key)
            .map_err(|_| -> AnyError { "data-plane authority certificate failed".into() })?;

        let key_digest = Sha256::digest(key.public_key_raw());
        let member_name = format!(
            "{MEMBER_NAME_PREFIX}{}{MEMBER_NAME_SUFFIX}",
            hex(key_digest.get(..MEMBER_NAME_DIGEST_BYTES).unwrap_or(&[]))
        );
        Ok(Self {
            issuer: Issuer::new(params, key),
            certificate_pem: certificate.pem(),
            member_name,
        })
    }

    /// Issue an ephemeral member certificate whose common name is `common_name`.
    fn issue_member(&self, common_name: &str) -> AnyResult<(String, Zeroizing<String>)> {
        let key = KeyPair::generate()
            .map_err(|_| -> AnyError { "data-plane member key generation failed".into() })?;
        let mut name = DistinguishedName::new();
        name.push(DnType::CommonName, common_name);
        name.push(DnType::OrganizationName, ORGANIZATION_NAME);
        let mut params = CertificateParams::default();
        params.distinguished_name = name;
        params.subject_alt_names =
            vec![SanType::DnsName(common_name.try_into().map_err(
                |_| -> AnyError { "data-plane member name is invalid".into() },
            )?)];
        params.is_ca = IsCa::ExplicitNoCa;
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        // Every member both accepts and dials, so the one certificate serves
        // as the TLS server and the TLS client identity.
        params.extended_key_usages = vec![
            ExtendedKeyUsagePurpose::ServerAuth,
            ExtendedKeyUsagePurpose::ClientAuth,
        ];
        set_unbounded_validity(&mut params);
        let certificate = params
            .signed_by(&key, &self.issuer)
            .map_err(|_| -> AnyError { "data-plane member certificate failed".into() })?;
        Ok((certificate.pem(), Zeroizing::new(key.serialize_pem())))
    }
}

/// Everything one process needs to join its cluster's Zenoh data plane.
pub struct DataPlaneTrust {
    authority_certificate_pem: String,
    member_certificate_pem: String,
    member_private_key_pem: Zeroizing<String>,
    member_name: String,
    scouting_port: u16,
}

impl fmt::Debug for DataPlaneTrust {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataPlaneTrust")
            .field("member_name", &self.member_name)
            .field("scouting_port", &self.scouting_port)
            .finish_non_exhaustive()
    }
}

impl DataPlaneTrust {
    /// Derive this process's trust material from the cluster key.
    ///
    /// # Errors
    /// Fails only if certificate generation fails; the message never contains
    /// key material.
    pub fn derive(cluster_key: &[u8; 32]) -> AnyResult<Self> {
        let authority = ClusterAuthority::derive(cluster_key)?;
        let member_name = authority.member_name.clone();
        Self::issued_by(&authority, cluster_key, &member_name)
    }

    /// Test-only: a member certificate chained to the real authority but
    /// carrying another name. It passes the TLS chain check exactly as a
    /// publicly trusted certificate would at a dialer, so it exercises the
    /// access-control layer that keeps such peers out.
    #[cfg(test)]
    pub(crate) fn derive_with_certificate_name(
        cluster_key: &[u8; 32],
        common_name: &str,
    ) -> AnyResult<Self> {
        let authority = ClusterAuthority::derive(cluster_key)?;
        Self::issued_by(&authority, cluster_key, common_name)
    }

    fn issued_by(
        authority: &ClusterAuthority,
        cluster_key: &[u8; 32],
        common_name: &str,
    ) -> AnyResult<Self> {
        let (member_certificate_pem, member_private_key_pem) =
            authority.issue_member(common_name)?;
        Ok(Self {
            authority_certificate_pem: authority.certificate_pem.clone(),
            member_certificate_pem,
            member_private_key_pem,
            member_name: authority.member_name.clone(),
            scouting_port: scouting_port(cluster_key),
        })
    }

    /// Test-only: the same trust on another scouting port, to show discovery
    /// separation holds on its own, independent of the TLS handshake.
    #[cfg(test)]
    pub(crate) const fn with_scouting_port(mut self, port: u16) -> Self {
        self.scouting_port = port;
        self
    }

    /// The common name every legitimate member certificate presents.
    #[must_use]
    pub fn member_name(&self) -> &str {
        &self.member_name
    }

    /// The multicast scouting group and port, `224.0.0.224:<port>`.
    #[must_use]
    pub fn scouting_address(&self) -> String {
        format!("{SCOUTING_GROUP}:{}", self.scouting_port)
    }

    /// PEM of the public authority certificate (the only trust anchor).
    #[must_use]
    pub fn authority_certificate_pem(&self) -> &str {
        &self.authority_certificate_pem
    }

    /// PEM of this process's member certificate.
    #[must_use]
    pub fn member_certificate_pem(&self) -> &str {
        &self.member_certificate_pem
    }

    /// PEM (PKCS#8) of this process's ephemeral member private key.
    pub(crate) fn member_private_key_pem(&self) -> &str {
        self.member_private_key_pem.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::{ClusterAuthority, scouting_port};
    use super::{DataPlaneTrust, SCOUTING_PORT_BASE, SCOUTING_PORT_SKIP, SCOUTING_PORT_SPAN};
    use rustls::RootCertStore;
    use rustls::client::verify_server_cert_signed_by_trust_anchor;
    use rustls::pki_types::{CertificateDer, UnixTime, pem::PemObject as _};
    use rustls::server::{ParsedCertificate, WebPkiClientVerifier};

    fn certificate(pem: &str) -> CertificateDer<'static> {
        CertificateDer::from_pem_slice(pem.as_bytes()).expect("valid certificate PEM")
    }

    fn roots(authority_pem: &str) -> RootCertStore {
        let mut store = RootCertStore::empty();
        store
            .add(certificate(authority_pem))
            .expect("authority is a usable trust anchor");
        store
    }

    /// Verify `member` as the TLS server chain under `authority`, the way a
    /// dialing peer does.
    fn server_chain_verifies(authority: &DataPlaneTrust, member: &DataPlaneTrust) -> bool {
        let member_certificate = certificate(member.member_certificate_pem());
        let parsed = ParsedCertificate::try_from(&member_certificate).expect("parse member");
        verify_server_cert_signed_by_trust_anchor(
            &parsed,
            &roots(authority.authority_certificate_pem()),
            &[],
            UnixTime::now(),
            rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .all,
        )
        .is_ok()
    }

    /// Verify `member` as a TLS client under `authority`, the way an accepting
    /// peer does with mutual TLS.
    fn client_chain_verifies(authority: &DataPlaneTrust, member: &DataPlaneTrust) -> bool {
        let verifier = WebPkiClientVerifier::builder_with_provider(
            roots(authority.authority_certificate_pem()).into(),
            rustls::crypto::ring::default_provider().into(),
        )
        .build()
        .expect("client verifier");
        verifier
            .verify_client_cert(
                &certificate(member.member_certificate_pem()),
                &[],
                UnixTime::now(),
            )
            .is_ok()
    }

    #[test]
    fn authority_derivation_is_deterministic() {
        let key = [7u8; 32];
        let first = ClusterAuthority::derive(&key).expect("derive");
        let second = ClusterAuthority::derive(&key).expect("derive");
        assert_eq!(first.certificate_pem, second.certificate_pem);
        assert_eq!(first.member_name, second.member_name);

        let one = DataPlaneTrust::derive(&key).expect("derive");
        let two = DataPlaneTrust::derive(&key).expect("derive");
        assert_eq!(
            one.authority_certificate_pem(),
            two.authority_certificate_pem()
        );
        assert_eq!(one.member_name(), two.member_name());
        assert_eq!(one.scouting_address(), two.scouting_address());
        // Member keys are per process, never derived from the cluster key.
        assert_ne!(one.member_private_key_pem(), two.member_private_key_pem());
        assert_ne!(one.member_certificate_pem(), two.member_certificate_pem());
    }

    #[test]
    fn same_key_members_trust_each_other_in_both_directions() {
        let key = [3u8; 32];
        let one = DataPlaneTrust::derive(&key).expect("derive");
        let two = DataPlaneTrust::derive(&key).expect("derive");
        assert!(server_chain_verifies(&one, &two));
        assert!(client_chain_verifies(&one, &two));
        assert!(server_chain_verifies(&two, &one));
        assert!(client_chain_verifies(&two, &one));
    }

    #[test]
    fn different_keys_yield_incompatible_trust() {
        let one = DataPlaneTrust::derive(&[1u8; 32]).expect("derive");
        let two = DataPlaneTrust::derive(&[2u8; 32]).expect("derive");
        assert_ne!(
            one.authority_certificate_pem(),
            two.authority_certificate_pem()
        );
        assert_ne!(one.member_name(), two.member_name());
        assert!(!server_chain_verifies(&one, &two));
        assert!(!client_chain_verifies(&one, &two));
        assert!(!server_chain_verifies(&two, &one));
        assert!(!client_chain_verifies(&two, &one));
    }

    #[test]
    fn member_name_is_unissuable_by_public_authorities() {
        let trust = DataPlaneTrust::derive(&[9u8; 32]).expect("derive");
        let name = trust.member_name();
        assert!(name.starts_with("skulk-dataplane-"));
        assert!(name.ends_with(".invalid"));
        // A DNS label is at most 63 characters.
        let label = name.trim_end_matches(".invalid");
        assert!(label.len() <= 63);
    }

    #[test]
    fn debug_output_never_contains_key_material() {
        let trust = DataPlaneTrust::derive(&[5u8; 32]).expect("derive");
        let rendered = format!("{trust:?}");
        assert!(!rendered.contains("PRIVATE KEY"));
        assert!(!rendered.contains(trust.member_private_key_pem()));
    }

    #[test]
    fn scouting_port_is_deterministic_and_in_range() {
        for byte in 0..=255u8 {
            let key = [byte; 32];
            let port = scouting_port(&key);
            assert_eq!(port, scouting_port(&key));
            assert!(port >= SCOUTING_PORT_BASE);
            assert!(port < SCOUTING_PORT_BASE + SCOUTING_PORT_SPAN);
            assert!(!SCOUTING_PORT_SKIP.contains(&port));
            // Never Zenoh's shared default discovery or listener port.
            assert_ne!(port, 7446);
            assert_ne!(port, 7447);
        }
    }

    #[test]
    fn scouting_ports_spread_across_keys() {
        let ports: std::collections::HashSet<u16> =
            (0..=255u8).map(|byte| scouting_port(&[byte; 32])).collect();
        // 256 keys over a 7000-port span: collisions are rare, so nearly all
        // ports are distinct. A constant or badly reduced hash would fail.
        assert!(ports.len() > 240, "only {} distinct ports", ports.len());
    }

    #[test]
    fn trust_address_uses_the_shared_link_local_group() {
        let trust = DataPlaneTrust::derive(&[4u8; 32]).expect("derive");
        let address = trust.scouting_address();
        assert!(address.starts_with("224.0.0.224:"));
        assert_eq!(
            address,
            format!("224.0.0.224:{}", scouting_port(&[4u8; 32]))
        );
    }
}
