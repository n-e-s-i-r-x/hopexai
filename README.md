# HopeXAI

HopeXAI is a persistent, multi-provider AI API gateway with one public model ID: **`hopexai`**. It exposes OpenAI-compatible Chat Completions and Responses APIs, a minimal key-generation website, private file-based upstream configuration, provider discovery, capability-aware routing, key rotation, circuit breakers, and bounded fallback.

This repository contains the gateway itself—not a frontend-only mock. A deterministic mock provider is used by the automated test suite. The ready-to-copy bundle includes a private local provider config; it is excluded from Git, Docker build context, and the Vercel frontend upload. No live paid-provider transaction has been run.

## What is implemented

- Minimal black-and-white public site at `/`; API-key generation requires no account or payment details.
- One public model from `GET /v1/models`: `hopexai`.
- Authenticated `POST /v1/chat/completions` and `POST /v1/responses`; non-streaming and SSE streaming.
- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages adapters. Provider/base URL/API-key additions use YAML only when they use one of these formats.
- Automatic, bounded endpoint exploration using OpenAPI/Swagger documents and standard candidates, `GET`, `HEAD`, `OPTIONS`, status codes, content types, headers, and a credential-free invalid-request check. OpenAPI paths are only retained for recognized API operations.
- Automatic model-list discovery, model metadata/capability/pricing parsing, TTL caching in the database, cross-worker discovery leases, and configuration-based endpoint/model/capability overrides.
- Strict capability routing for chat, streaming, vision, function calling, structured output, reasoning, and reasoning summaries. A request is eligible only for models explicitly verified to support every capability it uses. Audio, video, and document inputs are rejected because those public request paths are not implemented; audio, embeddings, image generation, moderation, reranking, and file-operation metadata may be discovered but are not public inference endpoints.
- Verified-free / paid / unknown pricing classification. A model name containing “free” has no effect. Free-only routing accepts only provider metadata or an explicit manual `pricing: free` override.
- Randomized provider/base/model selection with a strict context-first tier: eligible models with verified context windows of at least 1,000,000 tokens are preferred over every smaller or unknown-context model. When no 1M+ model is eligible, the largest known compatible context window is preferred. Existing priority/weight randomization, key rotation, health checks, and conversation affinity apply within the chosen context tier.
- Persistent provider/model/key health, key cooldowns, account-quota cooldowns, bounded retries, endpoint fallback, and recovery after cooldown.
- Reasoning effort (`minimal`, `low`, `medium`, `high`, `extra_high`, `maximum`), reasoning on/off, and optional **provider-exposed summaries only**. Private reasoning tokens are not fabricated or returned.
- Tool-call/function-call normalization; HopeXAI never executes tools.
- Encrypted, expiring Responses conversation state for `previous_response_id`; user-provided system/developer instructions are always lower priority than the server identity instruction.
- PostgreSQL persistence and Alembic initialization; SQLite is supported for local development and tests.
- API-key hashes (never plaintext), a persistent AES-GCM master key for encrypted provider keys/Responses state, private CLI revocation, abuse throttles, request-size limits, and private operational diagnostics.

### Important compatibility boundary

Discovery is intentionally strict: a model-list entry by itself does **not** prove that every model supports chat, vision, tools, reasoning, or streaming. If metadata does not prove a requested capability, HopeXAI records it as unknown and will not route a request requiring it. Use a verified `default_capabilities` declaration or a `manual_models` entry when provider documentation establishes the capability. Context priority also requires a known context value; if all eligible models have unknown context sizes, the gateway can only randomize among them and cannot guarantee a 1M+ selection. Pricing stays `unknown` unless the provider supplies explicit pricing metadata or you set it manually.

The initial discovery probes do not send provider credentials. A separate model-catalog GET may use a configured key only after the endpoint has been identified and the request is restricted to the exact configured origin; it never follows a cross-origin redirect and never logs the credential. This authenticated catalog read is necessary for providers that protect their model list.

## Quick start: local development

Requires Python 3.11+.

```bash
cd hopexai
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m app.cli setup
```

`setup` creates `private/hopexai.config.yaml` only if it is missing, creates a persistent `private/state/master.key`, and sets owner-only permissions. The ready-to-copy bundle already includes a populated private YAML file for local use. Upstream keys do **not** belong in browser code, static assets, or environment variables. Keep `private/hopexai.config.yaml` and `private/state/master.key` out of Git; both are covered by `.gitignore`. The separate Vercel frontend build excludes `private/`.

For a quick local SQLite run:

```bash
export HOPEXAI_HOME="$PWD"
export DATABASE_URL="sqlite+aiosqlite:///$PWD/var/hopexai.db"
export HOPEXAI_AUTO_CREATE_SCHEMA=true
python -m app.cli config validate
# Optional but recommended after adding plaintext keys:
python -m app.cli config encrypt-keys
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`. The public API base URL is the same origin plus `/v1`.

With an empty provider list, key generation and `GET /v1/models` work, but inference returns a generic service-unavailable error until an eligible upstream is configured. This avoids pretending that a mock provider is production infrastructure.

## Add a provider, base URL, key, or manual model

Edit only `private/hopexai.config.yaml`. The project includes a commented example at `config/hopexai.config.example.yaml`. A minimal OpenAI-compatible provider entry looks like this:

```yaml
schema_version: 1

defaults:
  default_max_output_tokens: 1024
  max_output_tokens_limit: 32768
  free_only: false
  max_attempts: 5
  discovery_ttl_seconds: 1800
  retry_ambiguous_failures: false

providers:
  - name: my_provider
    enabled: true
    priority: 10              # lower number wins the preferred tier
    selection_weight: 1.0
    api_format: auto           # auto | openai_chat | openai_responses | anthropic
    auth:
      mode: bearer             # bearer | header | query | none
      header_name: Authorization
      prefix: "Bearer "
      query_parameter: key
    reasoning:
      effort_parameter: reasoning_effort
      # Set these only when provider docs confirm the parameter exists:
      enabled_parameter: null
      summary_parameter: null
      disabled_effort_value: none
    base_urls:
      - url: https://api.example.com
        enabled: true
        selection_weight: 1.0
        api_keys:
          - YOUR_UPSTREAM_KEY
          - YOUR_SECOND_UPSTREAM_KEY
        model_discovery: true
        allow_private_network: false
        free_only: false
        include_models: null
        exclude_models: []
        default_capabilities:
          # Explicitly set only features confirmed for every model at this base.
          chat: true
          streaming: true
        endpoint_overrides: {}
        manual_models:
          - id: documented-chat-model
            enabled: true
            pricing: unknown   # free | paid | unknown
            context_window: 32768
            capabilities:
              chat: true
              streaming: true
              vision: false
              function_calling: true
              structured_output: true
              reasoning: false
              reasoning_summary: false
```

### Configuration behavior

- The filename is exactly `hopexai.config.yaml`; the default path is `private/hopexai.config.yaml`. `HOPEXAI_CONFIG` may point at a different **private** mount.
- Each provider can have multiple base URLs; each base can have multiple keys. Keys rotate only for unambiguously key-specific failures (for example, invalid credentials or an explicit per-key rate limit). An ambiguous `429`, account-wide quota, forbidden catalog read, or project limit cools down the base and does not try other keys to evade a shared provider quota. Set `key_rotation.rotate_on_rate_limit: false` to disable even explicit per-key rate-limit rotation.
- `endpoint_overrides` values are paths, not arbitrary URLs. An override is attempted first; discovered compatible paths remain available as fallback. The base URL itself may include a prefix such as `/v1`; discovery avoids duplicating version prefixes.
- `api_format: auto` uses discovered route/document evidence. For Anthropic-compatible providers, set `api_format: anthropic` and configure `auth.mode: header`, `header_name: x-api-key`, `prefix: ""` if discovery cannot identify the format.
- For provider formats outside the supported adapters, add a source-code adapter; merely adding a YAML entry cannot implement a new wire protocol.
- Use `manual_models` to include models absent from a catalog, disable individual models, define capability evidence, set a known context window, or explicitly classify pricing. Model IDs in `exclude_models` are always excluded. `include_models` restricts automatically discovered IDs.
- `default_capabilities` is an explicit assertion for every model at the base. Prefer per-model overrides where model features differ.
- If `free_only` is enabled globally or for a base, `paid` and `unknown` models are excluded. There is no fallback to paid/unknown models.
- Request timeouts, discovery TTLs, retry ceilings, health cooldowns, context-state retention, and output-token safety limits live under `defaults`.
- Invalid configuration reloads are rejected atomically: the process retains the last known-good snapshot. The file is polled periodically; no restart is needed. Write changes atomically when possible.

Validate and protect the file:

```bash
python -m app.cli config validate
python -m app.cli config encrypt-keys
```

`config encrypt-keys` replaces plaintext `api_keys` with AES-GCM `ENC[v1:...]` values using the persistent master key. The private YAML file and master key must be backed up and permission-restricted. Losing or replacing the master key makes encrypted values and stored Responses state unreadable and invalidates the HMACs used to authenticate all issued HopeXAI API keys, so keys would need to be reissued. Never commit either file.

## API usage

Generate a key at the website, then use it as a Bearer token:

```bash
curl https://YOUR-HOST/v1/models \
  -H 'Authorization: Bearer hx_YOUR_HOPEXAI_KEY'

curl https://YOUR-HOST/v1/chat/completions \
  -H 'Authorization: Bearer hx_YOUR_HOPEXAI_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"hopexai","messages":[{"role":"user","content":"Hello"}]}'
```

Streaming is enabled with `"stream": true`. The public API returns HopeXAI-owned model IDs and normalized responses; it does not return upstream model names, provider names, URLs, or keys.

Reasoning example:

```json
{
  "model": "hopexai",
  "reasoning_enabled": true,
  "reasoning_effort": "high",
  "include_reasoning": true,
  "messages": [{"role": "user", "content": "Solve this carefully."}]
}
```

`include_reasoning` requests only a summary that the selected upstream explicitly returns and supports. HopeXAI does not claim to provide hidden internal reasoning tokens. Providers with incompatible parameter names should be configured under `providers[].reasoning`.

Conversation affinity can be requested using `conversation_id` or `X-HopeXAI-Conversation-ID`. Initial provider selection is randomized; later requests with that identifier prefer the previous provider/base/model if it remains compatible and healthy. For `/v1/responses`, `previous_response_id` uses HopeXAI's encrypted state store, not an upstream opaque ID. Responses state defaults to a one-day TTL; `"store": false` disables saving the new response.

## Key administration

- Keys are cryptographically random and displayed once on the website.
- Only an HMAC hash, a short prefix, metadata, and usage counters are stored.
- Revoke a key privately by ID:

```bash
python -m app.cli keys list
python -m app.cli keys revoke KEY_UUID
```

Revocation is checked on every request and removes encrypted Responses history plus conversation-affinity state. In-flight requests also check key status before persisting new conversation state. There is no public key-management or provider dashboard.

Default abuse safeguards allow at most 3 generated keys per source IP per hour and 10 per day, plus 120 API requests per key per minute. These are operational anti-abuse controls, not paid subscription tiers; adjust them with `HOPEXAI_KEY_ISSUE_PER_HOUR`, `HOPEXAI_KEY_ISSUE_PER_DAY`, and `HOPEXAI_API_KEY_RPM` if deployment conditions require it.

## Diagnostics

All upstream diagnostics are private CLI operations:

```bash
python -m app.cli providers diagnose
python -m app.cli providers diagnose --force
python -m app.cli health reset
python -m app.cli db cleanup
```

Diagnostics include discovered routes/models, evidence-backed capabilities, pricing classification, and key health without printing key values. Request statistics contain IDs, route type, outcome, latency, and selected upstream identifiers in the private database; prompts and completions are not logged. Schedule `db cleanup` periodically to delete expired conversation state, discovery caches/leases, stale health rows, and old operational records.

## Production deployment (recommended: container + PostgreSQL)

1. Copy `.env.example` to `.env`. Set a strong PostgreSQL password and generate a separate random `HOPEXAI_VERCEL_PROXY_SECRET` (at least 32 characters). Keep `.env` out of GitHub.
2. Create `private/hopexai.config.yaml` from the example, add provider settings, validate it, and encrypt plaintext keys.
3. Ensure the private volume is writable/readable only by the container's service UID `10001` (or deliberately override the container UID). The master key and config must live on persistent storage.
4. Run `docker compose up -d --build`. The entrypoint applies Alembic migrations before starting two persistent Uvicorn workers.
5. Attach a TLS reverse proxy to the private Docker network. Do not publish port 8000 directly. Use `deploy-nginx.conf` as a reference; SSE proxy buffering is disabled and long read timeouts are configured.
6. Back up PostgreSQL and the private `/data` volume together. Use an external secrets/backup policy for `.env` and `private/state/master.key`.

The compose file intentionally exposes only an internal gateway port. Add a reverse-proxy service to `hopexai_internal` (subnet `172.30.0.0/16`) or adjust that subnet and `HOPEXAI_TRUSTED_PROXY_CIDRS` / Uvicorn forwarded-IP trust together. Do not set forwarded-IP trust to `*` on an internet-facing listener.

### Vercel frontend at `https://hopefreexai.vercel.app`

This repository includes a Vercel static-site build (`vercel.mjs` and `scripts/build-vercel.mjs`). The UI derives its displayed API base URL from its active origin, so when Vercel assigns `https://hopefreexai.vercel.app`, the UI displays `https://hopefreexai.vercel.app/v1`. Vercel rewrites `/api/keys`, `/v1/*`, `/readyz`, and `/healthz` to a separately deployed gateway using the Vercel **Project Environment Variable** `HOPEXAI_BACKEND_URL` (an HTTPS origin only, no path). API rewrite caching is explicitly disabled to prevent caching one-time keys or model responses. If the backend variable is unset, the static page still builds, but API/key requests are not connected.

Deployment sequence:

1. Push the cleaned source to GitHub; do not add the private YAML, master key, `.env`, or database files.
2. Deploy the FastAPI gateway using the preceding persistent-server steps and obtain its public HTTPS origin.
3. In the Vercel project's **Production** Environment Variables, set `HOPEXAI_BACKEND_URL` to that origin and set `HOPEXAI_VERCEL_PROXY_SECRET` to a random value of at least 32 characters. Put the exact same secret in the backend `.env`; Vercel overwrites a private header with it, and the gateway uses it to authenticate proxy traffic and safely read Vercel's forwarded visitor IP for key-issue rate limits.
4. On the gateway host set `HOPEXAI_PUBLIC_ORIGIN=https://hopefreexai.vercel.app`. The gateway then accepts key issuance only for that browser origin. These are deployment settings, not provider settings. Assign the requested production domain to the Vercel project in its dashboard; the code cannot reserve the domain. Redeploy Vercel after setting variables so the rewrite configuration is rebuilt.

The FastAPI gateway should run on a persistent server/container with PostgreSQL and a private persistent volume containing `hopexai.config.yaml` and `state/master.key`. Vercel hosts the minimal frontend and proxies API traffic; it does not receive the private provider config or master key. Follow the preceding steps to deploy the persistent gateway. The domain rewrite is same-origin for clients, but Vercel's proxy limits still apply to long-lived streaming requests; for strict, long-duration streaming, clients can use the gateway's own HTTPS origin directly.

## Safety and fallback policy

- Provider URLs must be absolute HTTP(S) URLs without embedded credentials, query strings, or fragments. Public endpoints must use HTTPS; plain HTTP is accepted only for explicitly opted-in private addresses. Private, loopback, and link-local destinations require an explicit base-URL opt-in. Metadata, multicast, reserved, unspecified, 6to4/Teredo, and other non-routable addresses remain blocked even with that opt-in. Every DNS answer is checked, and the validated IP is pinned at TCP connect while the original host is retained for TLS SNI and the HTTP `Host` header, preventing DNS-rebinding between validation and connection. Connections are not reused across validations. Redirects are limited to the same origin and revalidated as public; cross-origin redirects are not followed.
- Only configured origins are probed. Discovery does not send keys to docs or candidate endpoints. Authenticated catalog reads are made only to a discovered/manual model-list path on the configured origin.
- Ambiguous HTTP 5xx/read-timeout responses may already represent a completed billable operation. They are marked unhealthy but are **not replayed by default**. If a provider contractually honors an idempotency header, set `supports_idempotency: true` for that base and explicitly set `defaults.retry_ambiguous_failures: true` to permit idempotent fallback. A stream is never restarted after public output has been emitted.
- Context limits are estimated conservatively without depending on one provider's tokenizer. Known context windows are enforced; unknown windows remain unknown. HopeXAI never silently truncates or removes important messages.
- Capability metadata is evidence-based. If a catalog provides only IDs, add a verified manual capability declaration before routing.

## Verification

```bash
pytest -q
```

The suite exercises the website, one-time key creation, hashed storage, authentication/revocation, model-list privacy, discovery, key rotation, account-quota failover, manual endpoint fallback, free-model filtering, OpenAI Chat/Responses normalization (including incomplete-token endings), encrypted conversation continuity, tool calls, reasoning summaries, streaming, config reload safety, SSRF address rejection, DNS-IP pinning, and fail-closed outbound transport behavior. It runs a deterministic local mock provider; it does not claim a real paid-provider integration was tested without valid credentials.
