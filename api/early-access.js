import { put } from '@vercel/blob';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', ['POST']);
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { email } = req.body || {};

    if (typeof email !== 'string' || !email.trim()) {
      return res.status(400).json({ error: 'Invalid email' });
    }

    const normalized = email.trim().toLowerCase();
    const timestamp = new Date().toISOString();
    const safeEmail = normalized.replace(/[^a-z0-9]+/gi, '_').slice(0, 120);
    const pathname = `early-access/${timestamp}-${safeEmail}.json`;
    const body = JSON.stringify({ email: normalized, submittedAt: timestamp }) + '\n';

    const blob = await put(pathname, body, {
      // Your Blob store is configured as private, so the access must be 'private'.
      access: 'private',
      addRandomSuffix: true,
      contentType: 'application/json',
      token: process.env.BLOB_READ_WRITE_TOKEN,
    });

    return res.status(200).json({ ok: true, pathname: blob.pathname });
  } catch (error) {
    console.error('early-access error', error);
    return res.status(500).json({
      error: 'Internal Server Error',
      details: error instanceof Error ? error.message : String(error),
    });
  }
}

