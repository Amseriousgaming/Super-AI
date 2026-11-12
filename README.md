---
title: Hack404
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
short_description: 'Sustainable '
---
title: Personal Super AI
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
---

# Personal Super AI

A mobile-first frontend for an academic AI assistant covering elementary school through research level. It is plain HTML, CSS and JavaScript with no build step, no paid libraries and no subscription.

This repository is the **frontend only**. It does not contain an AI model and never invents answers. Every answer comes from a backend you run.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Application structure, dialogs and templates |
| `style.css` | Dark theme and responsive layout (tuned for Android) |
| `script.js` | Chat, history, images, settings and backend communication |
| `README.md` | This guide (the header above configures Hugging Face) |

Free libraries loaded from CDNs: marked (Markdown), DOMPurify (sanitising), KaTeX (equations), highlight.js (code), and Google Fonts.

## Features

- Chat with Markdown, tables, code highlighting and LaTeX equations (`$x^2$`, `$$...$$`, `\(...\)`, `\[...\]`)
- Image upload, camera capture, paste and drag-and-drop, with previews before sending
- Photos are resized on the device before upload
- Subject, academic level and answer mode selectors
- Conversation history in the sidebar, with search, delete and clear all (stored in this browser only)
- Copy answer, retry, stop generating, optional streaming
- Settings for backend URL, context length and image size
- Tools and models panel with placeholders for planned features

## Run locally

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. Without a backend, questions show a clear error instead of an answer.

## Deploy on Hugging Face Static Spaces

1. Create a new Space and choose **Static** as the SDK.
2. Upload `index.html`, `style.css`, `script.js` and `README.md` to the root.
3. Open the Space, go to **Settings** in the app, and set **Backend endpoint** to the full URL of your backend, for example `https://your-backend.example.com/api/chat`.

A Static Space only serves files, so `/api/chat` does not exist there. Your backend must be hosted elsewhere and must send CORS headers that allow your Space's origin.

## Backend contract

The app sends `POST` with `Content-Type: application/json`:

```json
{
  "question": "Solve 3x^2 - 12 = 0",
  "subject": "mathematics",
  "level": "secondary",
  "mode": "step-by-step",
  "model": "auto",
  "image": null,
  "images": [
    { "name": "photo.jpg", "mime_type": "image/jpeg", "data": "<base64>", "width": 1200, "height": 900 }
  ],
  "history": [
    { "role": "user", "content": "...", "has_images": false },
    { "role": "assistant", "content": "...", "has_images": false }
  ],
  "conversation_id": "abc123",
  "features": {},
  "stream": false,
  "client": { "name": "Personal Super AI", "version": "1.0.0", "locale": "en-US", "timezone": "UTC" }
}
```

Values:

- `subject`: `auto`, `mathematics`, `physics`, `chemistry`, `biology`, `history`, `geography`, `political-science`, `economics`, `english`, `computer-science`, `philosophy`, `astronomy`, `other`
- `level`: `auto`, `elementary`, `middle-school`, `secondary`, `higher-secondary`, `undergraduate`, `postgraduate`, `phd`, `research`
- `mode`: `detailed`, `step-by-step`, `short`, `deep-reasoning`, `research`
- `auto` means the backend should detect the subject or level from the question.
- `image` is the first image (or `null`); `images` holds all of them. `data` is base64 without a `data:` prefix.

### Response

Return JSON with the answer as Markdown:

```json
{ "answer": "Step 1: ..." }
```

The app also accepts `response`, `text`, `content`, or an OpenAI-style `choices[0].message.content`.

For errors, return a non-2xx status and `{ "error": "what went wrong" }`. The message is shown to the user.

### Streaming (optional)

Turn on **Stream answers** in Settings. Respond with `Content-Type: text/event-stream` and lines such as:

```
data: {"delta": "The derivative of "}
data: {"delta": "x^2 is 2x."}
data: [DONE]
```

Each `data:` line may be a JSON object with `delta`, `token`, `text` or `content`, or plain text. If the response is not an event stream, the app falls back to normal JSON.

## Security

- Never put API keys in these files. They are public once deployed.
- Keep model keys on your backend and read them from environment variables.
- Validate and limit request size on the backend (images can be several MB).
- Add authentication and rate limiting on the backend if the endpoint is public. The frontend has no question counter by design, and it makes no claim about how much access your backend allows.
- Answers are sanitised with DOMPurify before display.
- Conversations are stored in this browser's local storage, not on a server.

## Extending the app

`script.js` exposes `window.PersonalSuperAI`:

```js
// Add models your backend supports
PersonalSuperAI.registerModel({ id: 'my-model', label: 'My model' });

// Turn on a planned feature once the backend supports it
PersonalSuperAI.registerFeature({ id: 'math_verification', available: true });

// Change the request before it is sent (screen capture, PDFs, memory, web research)
PersonalSuperAI.hooks.beforeSend.push(async (payload, ctx) => {
  payload.extra = { /* ... */ };
  return payload;
});
```

Planned features, all listed as "Coming soon" in the Tools panel until enabled:

- Continuous screen understanding
- PDF understanding
- Long-term memory
- Multiple AI models
- Mathematical verification with SymPy
- Scientific calculations
- Web research

Most of these need backend work. The frontend already passes enabled features in the `features` field.

## Limits

- Up to 4 images per question, 25 MB each before resizing.
- Browser storage is limited. If it fills up, the oldest conversation is removed.
- Full-size images are kept in memory only, so retrying after a page reload sends the question without the image.


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference