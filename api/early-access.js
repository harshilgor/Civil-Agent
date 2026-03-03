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
    const line = `${timestamp},${normalized}\n`;

    await put('early-access.csv', line, {
      access: 'private',
      addRandomSuffix: false,
      contentType: 'text/csv',
      append: true,
      token: process.env.BLOB_READ_WRITE_TOKEN,
    });

    return res.status(200).json({ ok: true });
  } catch (error) {
    console.error('early-access error', error);
    return res.status(500).json({ error: 'Internal Server Error' });
  }
}

