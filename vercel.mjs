// Vercel serves the minimal frontend at hopefreexai.vercel.app and rewrites
// its public API paths to the separately hosted, persistent HopeXAI gateway.
// Set HOPEXAI_BACKEND_URL and HOPEXAI_VERCEL_PROXY_SECRET in Vercel's project
// environment after deploying the backend. Provider configuration remains in
// the backend's private YAML file.
const rawBackendUrl = process.env.HOPEXAI_BACKEND_URL?.trim();
let backendOrigin = null;

if (rawBackendUrl) {
  let backend;
  try {
    backend = new URL(rawBackendUrl);
  } catch {
    throw new Error('HOPEXAI_BACKEND_URL must be an absolute HTTPS origin.');
  }
  const localDevHost = ['localhost', '127.0.0.1', '[::1]'].includes(backend.hostname.toLowerCase());
  if (
    (backend.protocol !== 'https:' && !(backend.protocol === 'http:' && localDevHost)) ||
    !backend.hostname ||
    backend.username ||
    backend.password ||
    (backend.pathname !== '/' && backend.pathname !== '') ||
    backend.search ||
    backend.hash
  ) {
    throw new Error('HOPEXAI_BACKEND_URL must be an HTTPS origin without credentials, path, query, or fragment.');
  }
  if (backend.hostname.toLowerCase() === 'hopefreexai.vercel.app') {
    throw new Error('HOPEXAI_BACKEND_URL cannot point back to the Vercel frontend; that would create a rewrite loop.');
  }
  backendOrigin = backend.origin;
}

function proxyRoute(src, dest) {
  return {
    src,
    dest,
    transforms: [
      {
        type: 'request.headers',
        op: 'set',
        target: { key: 'x-hopexai-vercel-proxy-secret' },
        args: '$HOPEXAI_VERCEL_PROXY_SECRET',
        env: ['HOPEXAI_VERCEL_PROXY_SECRET'],
      },
    ],
  };
}

const routes = backendOrigin
  ? [
      proxyRoute('/api/keys', `${backendOrigin}/api/keys`),
      proxyRoute('/v1/(.*)', `${backendOrigin}/v1/$1`),
      proxyRoute('/readyz', `${backendOrigin}/readyz`),
      proxyRoute('/healthz', `${backendOrigin}/healthz`),
      { handle: 'filesystem' },
    ]
  : [];

export const config = {
  framework: null,
  installCommand: '',
  buildCommand: 'node scripts/build-vercel.mjs',
  outputDirectory: '.vercel-output',
  routes,
  headers: [
    {
      source: '/api/keys',
      headers: [{ key: 'x-vercel-enable-rewrite-caching', value: '0' }],
    },
    {
      source: '/v1/(.*)',
      headers: [{ key: 'x-vercel-enable-rewrite-caching', value: '0' }],
    },
    {
      source: '/readyz',
      headers: [{ key: 'x-vercel-enable-rewrite-caching', value: '0' }],
    },
    {
      source: '/healthz',
      headers: [{ key: 'x-vercel-enable-rewrite-caching', value: '0' }],
    },
  ],
};
