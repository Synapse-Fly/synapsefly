# Security policy

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Send it to us privately:

- DM [@SynapseFly](https://x.com/SynapseFly) on X, or
- open a [private security advisory](https://github.com/Synapse-Fly/synapsefly/security/advisories/new) on this repo.

Include what you did, what happened, and the affected URL or file. We aim to reply within 72 hours.

## Scope

In scope: this repository, `www.synapsefly.com` (the Next.js page) and `api.synapsefly.com` (the brain's
HTTP/WebSocket API). Out of scope: third-party services we only read (DexScreener, neuPrint), findings that require a
compromised device, and volumetric denial of service against the simulation loop.

This project holds no user accounts, no wallets and no personal data: the backend stores a session log and snapshot
PNGs of its own simulation. There is no token contract in this repository — **we will never DM you about a wallet,
a presale, an airdrop or a "connect wallet" page.** Treat any such message as a scam.
