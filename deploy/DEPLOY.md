# SynapseFly — Production Deploy

> Fill these in for your own box before you start (this file is public, so it names nothing):
>
> ```bash
> export VPS_IP=203.0.113.10              # your server
> export CADDY_CONTAINER=<your-caddy-container>   # the container that owns :80/:443
> export CADDYFILE=<your-Caddyfile>              # the config it loads
> ```

Architecture: **frontend on Vercel, backend (the brain) on your VPS**. The brain is one
always-on process (20 Hz sim loop + persistent WebSocket + in-RAM connectome), which is why
it cannot live on Vercel's serverless platform.

```
Browser ──https──> synapsefly.com          (Vercel: Next.js frontend)
        ──wss────> api.synapsefly.com  ──>  $CADDY_CONTAINER (existing Caddy, :443)
                                        ──>  synapsefly-backend:4000  (Docker, /opt/synapsefly)
```

The VPS (`$VPS_IP`) already runs many services and an existing **Caddy** container owns ports
80/443. We do NOT install a second web server. The backend
runs as its own isolated Docker stack that publishes **no host ports** and joins the existing
Caddy network so Caddy can reverse-proxy to it. Nothing else on the box is modified except a
single appended vhost block in the Caddyfile (see step 4).

---

## 1. DNS (at your registrar / DNS provider for synapsefly.com)

| Record | Name | Value |
|--------|------|-------|
| A      | `api`  (api.synapsefly.com) | `$VPS_IP` |
| (root/apex → Vercel) | `@` / `www` | per Vercel's "Add Domain" screen (A `76.76.21.21` or the CNAME Vercel shows) |

Wait for `api.synapsefly.com` to resolve to the VPS before step 4 (Caddy needs it for the TLS cert).

## 2. Copy the repo to the VPS (into a dedicated dir — touches nothing else)

From your Windows machine (PowerShell), from the repo root `C:\Users\USER\fly`:

```powershell
ssh root@$VPS_IP "mkdir -p /opt/synapsefly"
scp -r backend deploy root@${VPS_IP}:/opt/synapsefly/
```

Only `backend` and `deploy` are copied - the frontend is served by Vercel (step 6). If you want the optional
self-hosted frontend container instead, `scp -r frontend` as well and see step 5.

## 3. Configure env on the VPS

```bash
ssh root@$VPS_IP
cd /opt/synapsefly/deploy
cp synapsefly.env.example synapsefly.env
# the example ships FLY_MARKET=sim (no $SYNAPSE pair has listed yet): set FLY_TOKEN_ADDRESS / FLY_CHAIN
# and FLY_MARKET=dexscreener when it does
nano synapsefly.env
```

## 4. Add the API route to the existing Caddy (one appended block, nothing else changed)

```bash
cat /opt/synapsefly/deploy/Caddyfile.synapsefly.snippet >> "$CADDYFILE"
docker exec "$CADDY_CONTAINER" caddy reload --config /etc/caddy/Caddyfile
```

`caddy reload` is graceful and does not drop your other sites.

## 5. Build and start the backend

```bash
cd /opt/synapsefly/deploy
docker compose build backend
docker compose up -d backend
docker compose logs -f          # watch first boot (builds + caches the connectome, ~a few s)
```

The compose file also declares an **optional** `frontend` service behind the `selfhost` profile - an alternative to
step 6 for people who do not want Vercel. It needs `frontend/` copied to the VPS too
(`scp -r frontend root@${VPS_IP}:/opt/synapsefly/`), and then
`docker compose --profile selfhost build frontend && docker compose --profile selfhost up -d frontend`. Without the
profile flag compose ignores the service entirely, which is why `docker compose build backend` cannot fail on a
missing `frontend/` directory.

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
4. Deploy. Open https://synapsefly.com — the fly should be moving and the market panel
   should be live (labelled *simulated* until a `$SYNAPSE` pair is configured).

---

## Operations

```bash
# update after a code change (from Windows): re-copy backend/ then on the VPS:
cd /opt/synapsefly/deploy && docker compose build backend && docker compose up -d backend

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
# remove the appended vhost block from "$CADDYFILE", then:
docker exec "$CADDY_CONTAINER" caddy reload --config /etc/caddy/Caddyfile
rm -rf /opt/synapsefly
```
