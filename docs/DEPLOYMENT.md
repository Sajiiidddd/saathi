# Deploying Saathi

A deployment is one Python process (FastAPI + the voice pipeline), the static
client it serves, a retrieval index built at deploy time, and a database.
Audio flows peer-to-peer over WebRTC between the caller's browser and this
process — which drives most of the platform decisions below.

---

## 1. The two hard requirements

1. **HTTPS.** Browsers only grant microphone access in a secure context.
   `localhost` counts for development; any real host needs TLS.
2. **Reachable UDP.** WebRTC media negotiates its own UDP path directly to the
   server. An HTTPS-only platform (classic PaaS, most serverless) will
   connect, then produce silence. This rules out Azure App Service and
   similar HTTP-only frontends for the media path unless you add a TURN relay.

**Recommended target: a small Linux VM** (Azure B2s-class is plenty — the
heavy lifting happens in Azure Speech and the LLM API, not on this box) with:

- ports **80/443 TCP** open (Caddy terminates TLS, auto-provisions certificates)
- **UDP open** for WebRTC media (either the full ephemeral range 1024–65535
  from anywhere, or a restricted range if you configure ICE accordingly)
- a DNS name pointed at it (or `<ip>.nip.io` for a quick demo)

If the server must live behind strict NAT/firewalling, deploy a TURN relay
(coturn, or a hosted TURN service) and add it to the ICE servers on both
sides: the browser's `RTCPeerConnection` config in `client/index.html` and
`SmallWebRTCConnection(ice_servers=...)` server-side. Skip TURN unless you
actually need it — it adds a hop of latency.

---

## 2. Accounts and keys

| Service | Needed for | Where | Cost notes |
|---|---|---|---|
| **Azure Speech** | STT + TTS (required) | portal.azure.com → create *Speech service* | Free F0: 5h STT + 500k TTS chars/month, but **F0 has minimal concurrency** — move to S0 (pay-as-you-go, cents) for anything public |
| **LLM — pick one** | conversation + classifier + profiles | | |
| · Groq (default) | `SAATHI_LLM_PROVIDER=openai` | console.groq.com → API key | Free: 30 req/min, 14,400/day — fine for a demo; paid tier for real traffic |
| · Google AI Studio | `SAATHI_LLM_PROVIDER=gemini` | aistudio.google.com/apikey | Free tier is 20 req/day/model — **unusable without billing enabled** |
| · Azure OpenAI | `SAATHI_LLM_PROVIDER=azure` | ai.azure.com → deploy a model | Requires Pay-As-You-Go subscription (not available on Azure-for-Students) |
| Supabase (optional) | hosted Postgres | supabase.com → new project | Free tier fine |
| ElevenLabs (optional) | alternative TTS | elevenlabs.io | Free tier ~10k chars/month ≈ one session; stock voices only |

The region of the Speech resource matters: put it near your callers
(`centralindia` for India). The VM should live in the same region.

---

## 3. Environment variables — complete reference

Set these as real environment variables on the host (systemd `Environment=`,
Docker `-e`, or a `.env` file next to `server.py` — never committed).

```bash
# ---- required ----------------------------------------------------------
AZURE_SPEECH_API_KEY=            # Speech resource key
AZURE_SPEECH_REGION=centralindia # region SLUG from the endpoint URL

SAATHI_LLM_PROVIDER=openai       # openai | gemini | azure
OPENAI_API_KEY=gsk_...           # for provider=openai (Groq key works here)
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-120b

# ---- provider alternatives ---------------------------------------------
#GOOGLE_API_KEY=                 # provider=gemini
#SAATHI_LLM_MODEL=gemini-3.5-flash
#SAATHI_CLASSIFIER_MODEL=gemini-2.5-flash-lite
#AZURE_OPENAI_ENDPOINT=          # provider=azure
#AZURE_OPENAI_API_KEY=
#AZURE_OPENAI_DEPLOYMENT=

# ---- database -----------------------------------------------------------
# Empty = SQLite at logs/saathi.db. For Postgres/Supabase see §4.
#SAATHI_DATABASE_URL=postgresql://user:pass@host:5432/postgres

# ---- serving -------------------------------------------------------------
SAATHI_HOST=0.0.0.0              # bind publicly (Caddy proxies to it)
SAATHI_PORT=7860
SAATHI_ICE_SERVERS=stun:stun.l.google.com:19302   # REQUIRED on cloud VMs:
                                 # behind NAT the server's own candidates are
                                 # private IPs the browser can't reach
SAATHI_HELPLINE_REGION=IN        # helpline set shown + spoken (IN | GB | US)
SAATHI_LOG_DIR=logs

# ---- tuning (defaults are sane) ------------------------------------------
SAATHI_TTS_PROVIDER=azure
SAATHI_TTS_VOICE=en-IN-Aarti:DragonHDLatestNeural
SAATHI_TTS_LANGUAGE=en-IN
SAATHI_STT_LANGUAGES=en-IN,hi-IN
SAATHI_STT_SEGMENTATION_MS=350   # end-of-utterance silence window
SAATHI_VAD_STOP_SECS=0.4
SAATHI_LLM_TEMPERATURE=0.65
SAATHI_RAG_TOP_K=4
#SAATHI_TTS_RATE=1.0
#SAATHI_TTS_STYLE=
#ELEVENLABS_API_KEY=             # only if a language mode uses elevenlabs
```

Notes:
- The language picker overrides voice/STT per call; the values above are the
  defaults for the "default" mode and for anything the picker doesn't set.
- `SSL_CERT_FILE` is only needed behind TLS-intercepting corporate proxies —
  not on a normal cloud host.

---

## 4. Database: Supabase (Postgres)

SQLite works on a single box, but Postgres survives redeploys, lets the
dashboard read from anywhere, and is one env var away:

1. supabase.com → **New project** (pick the region closest to the VM).
2. Project → **Connect** → copy the **Session pooler** URI
   (`postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`).
   Use the session pooler, not the transaction pooler — the store holds one
   long-lived connection.
3. Set it as `SAATHI_DATABASE_URL`. **Tables create themselves on first boot**
   (`sessions`, `turns`, `utterances`, `latency`, `profiles`) — no SQL to run.
4. Verify: boot the server, then in Supabase's Table Editor confirm the five
   tables exist and a test call writes rows.

Operational notes:
- The app connects directly over Postgres protocol; Supabase Row Level
  Security applies to their client APIs, not this connection — treat the
  connection string itself as the secret.
- Free-tier projects pause after ~a week of inactivity; open the dashboard to
  wake it before a demo, or run a daily keepalive query.
- Back up before schema changes: Supabase → Database → Backups (paid), or a
  scheduled `pg_dump`.
- `turns` and `utterances` contain conversation text (never audio). If a
  deployment is public, say so on the page (the demo gate already does) and
  set a retention policy — a weekly
  `DELETE FROM utterances WHERE ts < extract(epoch from now() - interval '30 days')`
  is a reasonable start.

---

## 5. Build and run on the VM

```bash
# Ubuntu 22.04+ assumed
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv caddy

git clone <your-repo> saathi && cd saathi
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# keys: create .env from the reference in §3, then verify everything:
python3 scripts/check_keys.py

# knowledge base (downloads the ~130MB embedding model once):
.venv/bin/python scripts/fetch_classics.py
.venv/bin/python scripts/build_kb.py

.venv/bin/python server.py           # sanity check, then Ctrl-C
```

**Caddy** (`/etc/caddy/Caddyfile`) — TLS in two lines:

```
saathi.yourdomain.com {
    reverse_proxy localhost:7860
}
```

**systemd** (`/etc/systemd/system/saathi.service`):

```ini
[Unit]
Description=Saathi voice companion
After=network-online.target

[Service]
WorkingDirectory=/home/you/saathi
ExecStart=/home/you/saathi/.venv/bin/python server.py
Restart=on-failure
RestartSec=3
# Either point at the .env in WorkingDirectory (loaded automatically) or:
# EnvironmentFile=/etc/saathi/env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now saathi caddy
```

Azure NSG (or any cloud firewall): allow TCP 80, TCP 443, UDP 1024–65535 in.
Port 7860 stays closed to the world — only Caddy talks to it.

### Docker alternative

A `Dockerfile` ships in the repo root. WebRTC needs host networking, so this
path is Linux-hosts-only:

```bash
docker build -t saathi .
docker run --network host --env-file /etc/saathi/env --restart unless-stopped saathi
```

The image builds the KB index and bakes the embedding model at build time, so
cold starts are fast. (Docker Desktop on Mac/Windows has no host networking —
the container will run but callers won't get audio. Use the VM path there.)

---

## 6. Pre-launch checklist

- [ ] **Dial every number in `data/helplines.json` for your region.** These
      are shown to people who may be in crisis; a stale number is the single
      worst bug this product can have. (Status: IN numbers web-verified against
      telemanas.mohfw.gov.in and aasra.info on 2026-08-25 — phone verification
      deliberately deferred until an actual public launch.)
- [ ] `python3 scripts/check_keys.py` passes on the production host
- [ ] Azure Speech on **S0** (F0's concurrency will drop calls the moment two
      people connect)
- [ ] LLM quota matches expected traffic (Groq free = 30 req/min shared across
      ALL concurrent callers — roughly 3–4 simultaneous conversations)
- [ ] The demo-gate disclaimer renders on the landing page
- [ ] `SAATHI_HELPLINE_REGION` matches the audience
- [ ] Postgres reachable from the VM (`SELECT 1` via psql) if configured
- [ ] A full test call from a phone on mobile data (not the office network) —
      this catches NAT/UDP issues nothing else will
- [ ] Barge-in, one Hindi turn, and the "gym nutrition" honesty check pass
- [ ] `logs/` on a disk with headroom; logrotate or a cleanup cron for
      `*.jsonl` (the DB holds the durable copy)

## 7. Monitoring what you shipped

Everything observable is in the store. Useful queries (SQLite and Postgres
alike):

```sql
-- voice-to-voice p50 by day
SELECT date(started_at, 'unixepoch') d, count(*) sessions FROM sessions GROUP BY d;
SELECT voice_to_voice_secs FROM latency ORDER BY voice_to_voice_secs
  LIMIT 1 OFFSET (SELECT count(*)/2 FROM latency);

-- mode distribution (is she adapting, or stuck in guide?)
SELECT mode, count(*) FROM turns GROUP BY mode;

-- speculation hit rate (latency win)
SELECT avg(speculative_hit) FROM turns;

-- what do people ask that retrieval can't answer? (corpus gaps)
SELECT query, count(*) c FROM turns
  WHERE mode='guide' AND (sources='[]' OR sources IS NULL)
  GROUP BY query ORDER BY c DESC LIMIT 20;

-- degraded understander turns (provider trouble)
SELECT count(*) FROM turns WHERE degraded=1;
```

`logs/server.log` carries per-turn lines (`🔎 turn N: [mode] emotion · … ·
sources: …`) and every warning worth paging on: `understander degraded`,
`rag augmentation skipped`, `pipeline error`.

## 8. Cost picture at demo scale

Rough monthly numbers for light demo traffic (tens of sessions):

| Item | Cost |
|---|---|
| VM (Azure B2s) | ~$30–40, or free under student/startup credit |
| Azure Speech S0 | pennies at demo volume (TTS ~$16 per 1M chars) |
| Groq free tier | $0 within limits |
| Supabase free tier | $0 |
| Caddy TLS certs | $0 |

The system's marginal cost per conversation is effectively the TTS characters.
