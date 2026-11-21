# HopeXAI

HopeXAI is a single-project OpenAI-compatible gateway. It exposes one public model name, **`hopexai`**, and routes compatible Chat Completions and Responses requests to the providers configured in the active YAML file.

## Deploy the whole project to GitHub and Vercel

One Vercel project serves the landing page and runs the FastAPI API. There is no separately hosted backend, proxy service, or external database to provision. The active provider file is **`config/hopexai.config.yaml`**; edit that file directly before deploying. Vercel bundles it with the Python Function.

### 1. Download, extract, and edit the active config

1. Download the HopeXAI ZIP and extract it. Keep the project files together at the repository root; do not upload the ZIP as a single file.
2. Open **`config/hopexai.config.yaml`**. Replace the example provider URL and placeholder API key with values supported by your provider. Keep the provider's exact documented API base URL and model format.
3. The included defaults request **4,096 output tokens** and enforce a HopeXAI maximum of **8,192**. If an upstream rejects an output-token parameter or limit, HopeXAI can switch to a recognized alternative parameter or retry with progressively lower limits, up to ten bounded retries, but only for a definite pre-output validation rejection. Known smaller model context windows are handled before selection. Streaming retries happen before the first token is sent to the client; HopeXAI never restarts a stream after output has begun. An unrelated invalid request is not silently modified.
4. For OpenAI Chat-compatible providers, use `api_format: openai_chat`. For OpenAI Responses-compatible providers, use `api_format: openai_responses`. For Anthropic Messages-compatible providers, use `api_format: anthropic` and the provider's documented auth settings (often `x-api-key`, not Bearer auth). Only mark capabilities as supported when the provider confirms them.

**Keep provider credentials out of a public GitHub repository.** The safest setup is to keep a reference in the active config and store the actual secret in Vercel:

```yaml
        api_keys:
          - ${HOPEXAI_PROVIDER_API_KEY}
```

Then add `HOPEXAI_PROVIDER_API_KEY` as a server-side Vercel environment variable. HopeXAI resolves `${VARIABLE_NAME}` references in provider `api_keys` at runtime. You can also put a key directly in the YAML if you choose, but then it is part of your Git history and deployed source bundle; use a private repository and rotate the key if it is ever exposed. The ZIP contains only a placeholder, never a real provider key.

### 2. Push the extracted project to GitHub

Create a GitHub repository and upload the **contents** of the extracted folder. A private repository is recommended, especially if you put a provider key directly in the YAML. With Git installed, the commands from the project directory are:

```bash
git init -b main
git add .
git commit -m "Deploy HopeXAI"
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

If you change `api_keys` to `${HOPEXAI_PROVIDER_API_KEY}`, the actual provider key remains in Vercel rather than GitHub. Do not commit `.env` files, local databases, or a generated `private/master.key`.

### 3. Import the repository into Vercel and set environment variables

In Vercel, create a **New Project**, import the GitHub repository, and keep the project root pointed at the repository root. Vercel uses `pyproject.toml` (`app.main:app`) and `vercel.json` for the FastAPI Function and bundled static files.

In **Project → Settings → Environment Variables**, add this required variable for Production before deploying:

- **`HOPEXAI_MASTER_KEY`** — a stable random 32-byte signing key. Generate a URL-safe value locally with:

  ```bash
  python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
  ```

  Paste the generated value into Vercel; do not put it in the provider YAML or browser code. Keep it unchanged across deployments. Changing it invalidates every HopeXAI API key already issued. If using the safer `${HOPEXAI_PROVIDER_API_KEY}` config entry, also add **`HOPEXAI_PROVIDER_API_KEY`** as a Vercel environment variable. Add any other referenced provider key variables the same way.

Choose **Production** for the live deployment. Add variables for Preview only if you will test preview deployments, ideally with separate test credentials. After adding or changing an environment variable, redeploy: Vercel applies changed values to new deployments.

No `DATABASE_URL`, second backend URL, proxy secret, or `HOPEXAI_CONFIG_YAML` variable is required for this single-Vercel setup. `HOPEXAI_MASTER_KEY` must decode to exactly 32 bytes (64 hex characters or base64url); a `secrets.token_urlsafe(32)` value meets that requirement.

### 4. Deploy and use `hopefreexai.vercel.app`

Deploy the imported project. In Vercel's **Settings → Domains**, use or assign `hopefreexai.vercel.app` to this project if that domain is available in your Vercel account. The site and API share the same origin; the API base URL displayed by the page is `https://hopefreexai.vercel.app/v1`.

Open the deployed site and select **Generate API key**. Copy the key immediately; it is shown once. Use it as a Bearer token:

```bash
curl https://hopefreexai.vercel.app/v1/chat/completions \
  -H 'Authorization: Bearer hx_v1.YOUR_HOPEXAI_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"hopexai","messages":[{"role":"user","content":"Hello"}]}'
```

`GET /v1/models` returns only the public `hopexai` name. Requests are served by the same Vercel project as the landing page.

### 5. If key generation fails

The refreshed project includes a Vercel-mode key-issuance smoke test. Key issuance requires the stable `HOPEXAI_MASTER_KEY`; if the site was deployed without it, add the variable and redeploy. Also check that `config/hopexai.config.yaml` is present in the GitHub repository and that any `${...}` provider-key references have matching Vercel environment variables. For a continuing 500/503, open the latest Vercel deployment's **Function/Runtime Logs** and check the startup/configuration error. The page now shows the HTTP status, API error code when present, and request ID to help find the matching log entry; never send anyone your API key or provider credentials.

A 429 message means the per-instance key-issuance throttle was reached; wait and retry. These limits are best-effort and local to a warm function instance, not a global account quota.

The deployed site will not change just because a new ZIP exists: push the refreshed project to GitHub and let Vercel create a new deployment. The scroll and button styling fixes in this project take effect in that deployment.

## Configuration and local development

The single active provider file is `config/hopexai.config.yaml`. The same file is used locally and bundled for Vercel. Do not create a second example file or set `HOPEXAI_CONFIG_YAML` for Vercel. Run configuration validation before pushing:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m app.cli setup
python -m app.cli config validate
```

If the file references provider keys through `${VARIABLE_NAME}`, set those environment variables in the shell before validation or running the app. For local development, start the gateway with:

```bash
python -m app.cli db migrate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`. Local mode stores keys and state in its local database and a permission-restricted master key. It is different from Vercel's serverless key mode.

To run the automated suite:

```bash
pip install -e '.[test]'
pytest -q
make validate
```

Tests use mocked providers; they do not make a paid call or verify the live credentials in your Vercel account.

## Serverless behavior and security notes

- Vercel function filesystems are ephemeral. SQLite is only an instance-local cache for discovery and health; it is not durable database storage. HopeXAI keys are signed and work across instances when `HOPEXAI_MASTER_KEY` remains stable.
- In Vercel mode, individual API keys cannot be listed or revoked. Rotating the master key invalidates all issued HopeXAI keys. Key-generation and request throttles, provider health/cooldowns, and discovery caches are per warm function instance—not global quotas.
- Responses `previous_response_id` history is not persisted in Vercel mode. Send full conversation context in each request; Responses reports `store: false`.
- Ambiguous upstream failures are not replayed by default because a provider might already have processed a billable request. Output-token retries are limited to recognized validation errors, and streams are never switched after client output begins.
- Provider credentials are used by the server-side function and are not sent to the browser. Public API responses do not reveal upstream provider names, model IDs, URLs, or credentials. Do not treat instance-local rate limits as provider billing controls; set spending limits with your upstream provider.
