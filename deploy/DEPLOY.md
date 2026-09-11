# SynapseFly — Production Deploy

Architecture: **frontend on Vercel, backend (the brain) on your VPS**. The brain is one
always-on process (20 Hz sim loop + persistent WebSocket + in-RAM connectome), which is why
it cannot live on Vercel's serverless platform.

```
Browser ──https──> synapsefly.com          (Vercel: Next.js frontend)
        ──wss────> api.synapsefly.com  ──>  social-engine-proxy (existing Caddy, :443)
                                        ──>  synapsefly-backend:4000  (Docker, /opt/synapsefly)
```

The VPS (187.77.110.27) already runs many services and an existing **Caddy** container
(`social-engine-proxy`) owns ports 80/443. We do NOT install a second web server. The backend
runs as its own isolated Docker stack that publishes **no host ports** and joins the existing
Caddy network so Caddy can reverse-proxy to it. Nothing else on the box is modified except a
single appended vhost block in the Caddyfile (see step 4).

---

## 1. DNS (at your registrar / DNS provider for synapsefly.com)

| Record | Name | Value |
|--------|------|-------|
| A      | `api`  (api.synapsefly.com) | `187.77.110.27` |
| (root/apex → Vercel) | `@` / `www` | per Vercel's "Add Domain" screen (A `76.76.21.21` or the CNAME Vercel shows) |

Wait for `api.synapsefly.com` to resolve to the VPS before step 4 (Caddy needs it for the TLS cert).

## 2. Copy the repo to the VPS (into a dedicated dir — touches nothing else)

From your Windows machine (PowerShell), from the repo root `C:\Users\USER\fly`:

```powershell
ssh root@187.77.110.27 "mkdir -p /opt/synapsefly"
scp -r backend deploy root@187.77.110.27:/opt/synapsefly/
```

## 3. Configure env on the VPS

```bash
ssh root@187.77.110.27
cd /opt/synapsefly/deploy
cp synapsefly.env.example synapsefly.env
# edit synapsefly.env if needed (market CA is already set to your CASHCAT token)
nano synapsefly.env
```

## 4. Add the API route to the existing Caddy (one appended block, nothing else changed)

```bash
cat /opt/synapsefly/deploy/Caddyfile.synapsefly.snippet >> /root/crypto-automation/Caddyfile.vps
docker exec social-engine-proxy caddy reload --config /etc/caddy/Caddyfile
```

`caddy reload` is graceful and does not drop your other sites.

## 5. Build and start the backend

```bash
cd /opt/synapsefly/deploy
docker compose build
docker compose up -d
docker compose logs -f          # watch first boot (builds + caches the connectome, ~a few s)
```

Verify (on the VPS):

```bash
docker exec synapsefly-backend python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:4000/api/health').read().decode())"
```

Verify (from anywhere, once DNS + Caddy are live):

```bash
curl https://api.synapsefly.com/api/health
```

## 6. Frontend on Vercel

1. Import the repo in Vercel, set **Root Directory = `frontend`** (framework auto-detected: Next.js).
2. Project → Settings → Environment Variables (Production):
   - `NEXT_PUBLIC_WS_URL = wss://api.synapsefly.com/ws`
   - `NEXT_PUBLIC_API_URL = https://api.synapsefly.com`
3. Add domain `synapsefly.com` (and `www`) under Project → Domains.
4. Deploy. Open https://synapsefly.com — the fly should be moving and the market panel should
   show live CASHCAT data.

---

## Operations

```bash
# update after a code change (from Windows): re-copy backend/ then on the VPS:
cd /opt/synapsefly/deploy && docker compose build && docker compose up -d

docker compose logs -f            # logs
docker compose restart backend    # restart
docker compose down               # stop (frees nothing of yours; only this stack)
```

Switch the fly's diet later (no rebuild — just edit env + `docker compose up -d`):
- different token: `FLY_TOKEN_ADDRESS` / `FLY_CHAIN`
- real tweets: `FLY_LLM=anthropic` + `ANTHROPIC_API_KEY`, then `FLY_X=post` + the four `X_*` keys
- bigger brain: `FLY_N_NEURONS=166700` (bump the container to ~2 GB RAM)

## Rollback / clean removal (leaves the rest of the box untouched)

```bash
cd /opt/synapsefly/deploy && docker compose down -v
# remove the appended vhost block from /root/crypto-automation/Caddyfile.vps, then:
docker exec social-engine-proxy caddy reload --config /etc/caddy/Caddyfile
rm -rf /opt/synapsefly
```
