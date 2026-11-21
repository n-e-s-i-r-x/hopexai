(() => {
  const baseUrl = `${window.location.origin}/v1`;
  const baseNode = document.getElementById('base-url');
  const feedback = document.getElementById('feedback');
  const generateButton = document.getElementById('generate-key');
  const result = document.getElementById('key-result');
  const keyNode = document.getElementById('api-key');
  const copyBaseButton = document.getElementById('copy-base');
  const copyKeyButton = document.getElementById('copy-key');

  baseNode.textContent = baseUrl;

  function setFeedback(message, error = false) {
    feedback.textContent = message;
    feedback.classList.toggle('error', error);
  }

  async function copyText(value, label) {
    try {
      await navigator.clipboard.writeText(value);
      setFeedback(`${label} copied.`);
    } catch (_) {
      const field = document.createElement('textarea');
      field.value = value;
      field.setAttribute('readonly', '');
      field.style.position = 'fixed';
      field.style.opacity = '0';
      document.body.appendChild(field);
      field.select();
      const copied = document.execCommand('copy');
      field.remove();
      if (copied) setFeedback(`${label} copied.`);
      else setFeedback('Copy was blocked by this browser. Select and copy the value.', true);
    }
  }

  copyBaseButton.addEventListener('click', () => copyText(baseUrl, 'Base URL'));
  copyKeyButton.addEventListener('click', () => copyText(keyNode.textContent, 'API key'));

  generateButton.addEventListener('click', async () => {
    generateButton.disabled = true;
    generateButton.textContent = 'Generating…';
    setFeedback('');
    try {
      const response = await fetch('/api/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: '{}',
        cache: 'no-store',
        credentials: 'same-origin',
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.error?.message || 'A key could not be generated. Please try again later.');
      }
      if (typeof data.api_key !== 'string' || !data.api_key.startsWith('hx_')) {
        throw new Error('The server returned an invalid key response.');
      }
      keyNode.textContent = data.api_key;
      result.hidden = false;
      setFeedback('API key generated. Copy it now; it will not be shown again.');
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : 'A key could not be generated.', true);
    } finally {
      generateButton.disabled = false;
      generateButton.textContent = 'Generate API key';
    }
  });
})();
