/* ==========================================================================
   Personal Super AI - script.js
   Frontend only. All answers come from your secure backend (POST /api/chat).
   No API keys, no hardcoded answers.

   Sections
     1. Configuration (subjects, levels, modes, models, future features)
     2. Storage
     3. Utilities
     4. Markdown + math rendering
     5. Image handling
     6. AI client (backend communication)
     7. UI rendering
     8. Chat flow
     9. Settings, tools, sidebar
    10. Public extension API and startup
   ========================================================================== */
(() => {
  'use strict';

  /* ======================================================================
     1. CONFIGURATION
     ====================================================================== */
  const APP = { name: 'Personal Super AI', version: '1.0.0', prefix: 'psai.v1.' };

  // Add a subject here and it appears in the selector and on the welcome screen.
  const SUBJECTS = [
    { id: 'auto',              label: 'Auto',              hue: 0,   hint: '' },
    { id: 'mathematics',       label: 'Mathematics',       hue: 230, hint: 'Solve 3x^2 - 12 = 0 and explain each step' },
    { id: 'physics',           label: 'Physics',           hue: 200, hint: 'Why does a satellite stay in orbit?' },
    { id: 'chemistry',         label: 'Chemistry',         hue: 150, hint: 'Balance Fe + O2 -> Fe2O3' },
    { id: 'biology',           label: 'Biology',           hue: 125, hint: 'How does CRISPR-Cas9 edit a gene?' },
    { id: 'history',           label: 'History',           hue: 35,  hint: 'What caused the fall of the Western Roman Empire?' },
    { id: 'geography',         label: 'Geography',         hue: 95,  hint: 'How do monsoons form?' },
    { id: 'political-science', label: 'Political Science', hue: 350, hint: 'Compare presidential and parliamentary systems' },
    { id: 'economics',         label: 'Economics',         hue: 50,  hint: 'What happens to prices when demand is inelastic?' },
    { id: 'english',           label: 'English',           hue: 290, hint: 'Analyse the theme of ambition in Macbeth' },
    { id: 'computer-science',  label: 'Computer Science',  hue: 175, hint: 'Explain Dijkstra\'s algorithm with an example' },
    { id: 'philosophy',        label: 'Philosophy',        hue: 265, hint: 'What is Kant\'s categorical imperative?' },
    { id: 'astronomy',         label: 'Astronomy',         hue: 215, hint: 'How do astronomers measure the distance to a galaxy?' },
    { id: 'other',             label: 'Other',             hue: 220, hint: '' }
  ];

  const LEVELS = [
    { id: 'auto',             label: 'Auto' },
    { id: 'elementary',       label: 'Elementary' },
    { id: 'middle-school',    label: 'Middle school' },
    { id: 'secondary',        label: 'Secondary' },
    { id: 'higher-secondary', label: 'Higher secondary' },
    { id: 'undergraduate',    label: 'Undergraduate' },
    { id: 'postgraduate',     label: 'Postgraduate' },
    { id: 'phd',              label: 'PhD' },
    { id: 'research',         label: 'Research level' }
  ];

  const MODES = [
    { id: 'detailed',       label: 'Detailed' },
    { id: 'step-by-step',   label: 'Step by step' },
    { id: 'short',          label: 'Short answer' },
    { id: 'deep-reasoning', label: 'Deep reasoning' },
    { id: 'research',       label: 'Research' }
  ];

  // The backend decides which models exist. Register more with PersonalSuperAI.registerModel().
  const MODELS = [{ id: 'auto', label: 'Backend default' }];

  // Planned capabilities. `available: false` keeps the switch disabled.
  // When your backend supports one, call PersonalSuperAI.registerFeature({ id, available: true }).
  const FEATURES = [
    { id: 'screen_understanding', name: 'Screen understanding',  desc: 'Follow what is on your screen while you work.', available: false },
    { id: 'pdf_understanding',    name: 'PDF understanding',     desc: 'Read and answer questions about PDF files.', available: false },
    { id: 'long_term_memory',     name: 'Long-term memory',      desc: 'Remember your courses and preferences across sessions.', available: false },
    { id: 'math_verification',    name: 'Math verification',     desc: 'Check results with SymPy on the backend.', available: false },
    { id: 'scientific_calc',      name: 'Scientific calculations', desc: 'Run exact numeric and unit calculations.', available: false },
    { id: 'web_research',         name: 'Web research',          desc: 'Look up sources and cite them.', available: false }
  ];

  const DEFAULT_SETTINGS = {
    endpoint: '/api/chat',
    historyLimit: 20,
    imageMaxDim: 1600,
    stream: false,
    features: {}
  };

  const LIMITS = {
    maxImages: 4,
    maxFileBytes: 25 * 1024 * 1024,
    thumbDim: 240,
    requestTimeoutMs: 300000,
    maxConversations: 100
  };

  const DEFAULT_IMAGE_PROMPT = 'Solve or explain what is shown in the attached image.';

  /* ======================================================================
     2. STORAGE (localStorage, always wrapped: it can be blocked or full)
     ====================================================================== */
  const Store = {
    get(key, fallback) {
      try {
        const raw = localStorage.getItem(APP.prefix + key);
        return raw === null ? fallback : JSON.parse(raw);
      } catch { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(APP.prefix + key, JSON.stringify(value)); return true; }
      catch { return false; }
    },
    remove(key) {
      try { localStorage.removeItem(APP.prefix + key); } catch { /* ignore */ }
    }
  };

  /* ======================================================================
     3. UTILITIES AND STATE
     ====================================================================== */
  const el = {};
  [
    'app', 'sidebar', 'sidebarCloseBtn', 'sidebarOpenBtn', 'sidebarScrim', 'newChatBtn', 'newChatTopBtn',
    'historySearch', 'conversationList', 'historyEmpty', 'settingsBtn', 'clearAllBtn', 'conversationTitle',
    'stage', 'welcome', 'subjectGrid', 'messages', 'jumpBtn', 'subjectSelect', 'levelSelect', 'modeSelect',
    'toolsBtn', 'attachmentTray', 'composerForm', 'attachBtn', 'cameraBtn', 'questionInput', 'sendBtn',
    'stopBtn', 'fileInput', 'cameraInput', 'settingsDialog', 'settingsForm', 'endpointInput',
    'historyLimitInput', 'imageSizeInput', 'streamToggle', 'settingsResetBtn', 'toolsDialog', 'modelSelect',
    'toolList', 'toasts', 'lightbox', 'lightboxImg'
  ].forEach((id) => { el[id] = document.getElementById(id); });

  const state = {
    settings: { ...DEFAULT_SETTINGS },
    prefs: { subject: 'auto', level: 'auto', mode: 'detailed', model: 'auto' },
    conversations: [],
    activeId: null,
    attachments: [],            // images waiting to be sent
    sessionImages: new Map(),   // messageId -> full-size images (memory only, never stored)
    historyFilter: '',
    busy: false,
    controller: null
  };

  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  const tpl = (id) => document.getElementById(id).content.firstElementChild.cloneNode(true);
  const labelOf = (list, id) => (list.find((x) => x.id === id) || {}).label || '';
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const clamp = (n, min, max) => Math.min(max, Math.max(min, n));
  const isDesktop = () => window.matchMedia('(min-width: 900px)').matches;

  function toast(message, kind = 'info') {
    while (el.toasts.children.length >= 3) el.toasts.firstChild.remove();
    const t = document.createElement('div');
    t.className = 'toast toast--' + kind;
    t.textContent = message;
    el.toasts.appendChild(t);
    setTimeout(() => {
      t.classList.add('is-leaving');
      setTimeout(() => t.remove(), 260);
    }, 3200);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.cssText = 'position:fixed;opacity:0;top:0;left:0';
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch { /* ignore */ }
      ta.remove();
      return ok;
    }
  }

  function timeLabel(ts) {
    const d = new Date(ts);
    return d.toDateString() === new Date().toDateString()
      ? d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
      : d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }

  /* ======================================================================
     4. MARKDOWN + MATH RENDERING
     Pipeline: protect code -> protect math -> marked -> DOMPurify -> KaTeX.
     If a CDN library fails to load, answers fall back to safe plain text.
     ====================================================================== */
  const Markdown = (() => {
    let initialised = false;

    function init() {
      if (initialised) return;
      initialised = true;
      if (window.marked) window.marked.setOptions({ gfm: true, breaks: false });
      if (window.hljs) window.hljs.configure({ ignoreUnescapedHTML: true });
      if (window.DOMPurify) {
        window.DOMPurify.addHook('afterSanitizeAttributes', (node) => {
          if (node.tagName === 'A') {
            node.setAttribute('target', '_blank');
            node.setAttribute('rel', 'noopener noreferrer');
          }
        });
      }
    }

    const MATH_PATTERNS = [
      { re: /\$\$([\s\S]+?)\$\$/g, display: true },
      { re: /\\\[([\s\S]+?)\\\]/g, display: true },
      { re: /\\\(([\s\S]+?)\\\)/g, display: false },
      // Inline $...$ : no space just inside the dollars, and not followed by a digit (avoids "$5 and $10").
      { re: /(?<![\\$])\$(?!\s)((?:\\.|[^$\\\n])+?)(?<!\s)\$(?!\d)/g, display: false }
    ];

    function renderMath(tex, display) {
      if (window.katex) {
        try {
          return window.katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore', output: 'htmlAndMathml' });
        } catch { /* fall through */ }
      }
      return escapeHtml(display ? '$$' + tex + '$$' : '$' + tex + '$');
    }

    function render(source) {
      init();
      if (!window.marked || !window.DOMPurify) return '<p class="plain">' + escapeHtml(source) + '</p>';

      const codes = [];
      const maths = [];

      // 1. Protect fenced code (even unclosed, while streaming) and inline code.
      let text = source.replace(/(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]+`)/g, (m) => {
        codes.push(m);
        return 'PSAICODE' + (codes.length - 1) + 'END';
      });

      // 2. Protect math so Markdown cannot mangle underscores, backslashes or asterisks.
      for (const { re, display } of MATH_PATTERNS) {
        text = text.replace(re, (m, tex) => {
          maths.push({ tex: tex.trim(), display });
          const token = 'PSAIMATH' + (maths.length - 1) + 'END';
          return display ? '\n\n' + token + '\n\n' : token;
        });
      }

      // 3. Restore code, parse Markdown, sanitise.
      text = text.replace(/PSAICODE(\d+)END/g, (m, i) => codes[+i]);
      const html = window.DOMPurify.sanitize(window.marked.parse(text), { USE_PROFILES: { html: true } });

      // 4. Insert KaTeX output (generated locally, trust disabled).
      return html.replace(/PSAIMATH(\d+)END/g, (m, i) => renderMath(maths[+i].tex, maths[+i].display));
    }

    // Post-render polish: scrollable tables, syntax highlighting, copy buttons on code.
    function enhance(root, final) {
      root.querySelectorAll('table').forEach((t) => {
        if (t.parentElement.classList.contains('table-wrap')) return;
        const wrap = document.createElement('div');
        wrap.className = 'table-wrap';
        t.replaceWith(wrap);
        wrap.appendChild(t);
      });
      root.querySelectorAll('pre').forEach((pre) => {
        const code = pre.querySelector('code');
        if (final && code && window.hljs) {
          try { window.hljs.highlightElement(code); } catch { /* ignore */ }
        }
        if (!pre.querySelector('.code-copy')) {
          const b = document.createElement('button');
          b.type = 'button';
          b.className = 'code-copy';
          b.textContent = 'Copy';
          b.setAttribute('aria-label', 'Copy code');
          pre.insertBefore(b, pre.firstChild);
        }
      });
    }

    return { render, enhance };
  })();

  /* ======================================================================
     5. IMAGE HANDLING
     Photos are resized on the device and converted to JPEG base64 so uploads
     stay small on mobile data. Full images live in memory only; a small
     thumbnail is saved with the conversation.
     ====================================================================== */
  const Images = {
    async decode(file) {
      if (window.createImageBitmap) {
        try { return await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch { /* fall back */ }
      }
      return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
        img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('decode')); };
        img.src = url;
      });
    },

    fit(w, h, max) {
      const scale = Math.min(1, max / Math.max(w, h));
      return { w: Math.max(1, Math.round(w * scale)), h: Math.max(1, Math.round(h * scale)) };
    },

    draw(source, w, h, quality) {
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';              // flatten transparency so screenshots stay readable
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(source, 0, 0, w, h);
      return canvas.toDataURL('image/jpeg', quality);
    },

    async fromFile(file) {
      if (!file.type.startsWith('image/')) throw new Error('not-image');
      if (file.size > LIMITS.maxFileBytes) throw new Error('too-large');
      const bmp = await Images.decode(file);
      const full = Images.fit(bmp.width, bmp.height, state.settings.imageMaxDim);
      const thumb = Images.fit(bmp.width, bmp.height, LIMITS.thumbDim);
      const dataUrl = Images.draw(bmp, full.w, full.h, 0.92);
      const thumbUrl = Images.draw(bmp, thumb.w, thumb.h, 0.7);
      if (bmp.close) bmp.close();
      return {
        id: uid(),
        name: file.name || 'image',
        mime_type: 'image/jpeg',
        data: dataUrl.split(',')[1],        // base64 for the backend
        dataUrl,                            // for local preview
        thumb: thumbUrl,
        width: full.w,
        height: full.h
      };
    }
  };

  async function addFiles(fileList) {
    const files = Array.from(fileList || []).filter((f) => f && f.type && f.type.startsWith('image/'));
    if (!files.length) return;
    const room = LIMITS.maxImages - state.attachments.length;
    if (room <= 0) { toast('You can attach up to ' + LIMITS.maxImages + ' images per question.', 'error'); return; }
    if (files.length > room) toast('Only ' + LIMITS.maxImages + ' images fit in one question.');
    for (const file of files.slice(0, room)) {
      try {
        state.attachments.push(await Images.fromFile(file));
      } catch (err) {
        toast(err.message === 'too-large' ? 'That image is larger than 25 MB.' : 'Could not read that image. Try a JPEG or PNG.', 'error');
      }
    }
    renderAttachments();
    updateComposerState();
  }

  function renderAttachments() {
    el.attachmentTray.replaceChildren();
    state.attachments.forEach((att) => {
      const node = tpl('attachmentTemplate');
      const img = node.querySelector('img');
      img.src = att.thumb;
      img.alt = 'Attached image: ' + att.name;
      img.addEventListener('click', () => openLightbox(att.dataUrl));
      node.querySelector('.attachment__remove').addEventListener('click', () => {
        state.attachments = state.attachments.filter((a) => a.id !== att.id);
        renderAttachments();
        updateComposerState();
      });
      el.attachmentTray.appendChild(node);
    });
    el.attachmentTray.hidden = state.attachments.length === 0;
  }

  function openLightbox(src) {
    el.lightboxImg.src = src;
    el.lightbox.showModal();
  }

  /* ======================================================================
     6. AI CLIENT
     Request  (POST, JSON):
       { question, subject, level, mode, model,
         image, images[], history[], conversation_id, features{}, stream, client{} }
     Response (JSON): { "answer": "markdown text" }
       or, when streaming, a text/event-stream with `data:` lines.
     ====================================================================== */
  class ApiError extends Error {
    constructor(message, status) { super(message); this.name = 'ApiError'; this.status = status; }
  }

  const AIClient = {
    buildPayload({ question, meta, history, images, conversationId }) {
      const out = (i) => ({ name: i.name, mime_type: i.mime_type, data: i.data, width: i.width, height: i.height });
      const features = {};
      FEATURES.forEach((f) => { if (f.available && state.settings.features[f.id]) features[f.id] = true; });
      return {
        question,
        subject: meta.subject,
        level: meta.level,
        mode: meta.mode,
        model: state.prefs.model,
        image: images[0] ? out(images[0]) : null,
        images: images.map(out),
        history,
        conversation_id: conversationId,
        features,
        stream: !!state.settings.stream,
        client: {
          name: APP.name,
          version: APP.version,
          locale: navigator.language,
          timezone: (Intl.DateTimeFormat().resolvedOptions() || {}).timeZone || ''
        }
      };
    },

    async send(payload, { signal, onToken }) {
      const { endpoint, stream } = state.settings;
      const inner = new AbortController();
      let timedOut = false;
      const relay = () => inner.abort();
      signal.addEventListener('abort', relay);
      const timer = setTimeout(() => { timedOut = true; inner.abort(); }, LIMITS.requestTimeoutMs);

      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': stream ? 'text/event-stream, application/json' : 'application/json'
          },
          body: JSON.stringify(payload),
          signal: inner.signal,
          credentials: 'same-origin'
        });
        if (!res.ok) throw await AIClient.httpError(res);

        const type = res.headers.get('content-type') || '';
        if (stream && type.includes('text/event-stream') && res.body) {
          clearTimeout(timer);
          return await AIClient.readStream(res, onToken);
        }
        const data = type.includes('application/json') ? await res.json() : await res.text();
        return AIClient.extractAnswer(data);
      } catch (err) {
        if (timedOut) throw new ApiError('The backend took too long to respond. Try again, or ask a shorter question.');
        throw err;
      } finally {
        clearTimeout(timer);
        signal.removeEventListener('abort', relay);
      }
    },

    async httpError(res) {
      let detail = '';
      try {
        const type = res.headers.get('content-type') || '';
        if (type.includes('application/json')) {
          const j = await res.json();
          detail = (j.error && (j.error.message || j.error)) || j.detail || j.message || '';
          if (typeof detail !== 'string') detail = JSON.stringify(detail);
        }
      } catch { /* ignore */ }
      if (!detail && (res.status === 404 || res.status === 405)) {
        detail = 'No backend answers at ' + state.settings.endpoint + '. A static page cannot answer questions by itself: set your backend URL in Settings.';
      }
      return new ApiError('The backend returned an error (' + res.status + ')' + (detail ? ': ' + detail : '.'), res.status);
    },

    extractAnswer(data) {
      let answer = data;
      if (data && typeof data === 'object') {
        answer = data.answer ?? data.response ?? data.text ?? data.content
          ?? (data.message && (data.message.content ?? data.message))
          ?? (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content);
      }
      if (typeof answer !== 'string' || !answer.trim()) {
        throw new ApiError('The backend replied without an answer. Expected JSON like {"answer": "..."}.');
      }
      return answer;
    },

    async readStream(res, onToken) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let full = '';

      const handle = (data) => {
        if (data === '[DONE]') return true;
        let piece = data;
        try {
          const j = JSON.parse(data);
          if (typeof j === 'string') piece = j;
          else if (j && typeof j === 'object') {
            if (j.error) throw new ApiError(typeof j.error === 'string' ? j.error : (j.error.message || 'The backend stopped the answer.'));
            piece = j.delta ?? j.token ?? j.text ?? j.content
              ?? (j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content) ?? '';
          }
        } catch (e) {
          if (e instanceof ApiError) throw e;   // otherwise: plain-text token
        }
        if (piece) { full += piece; onToken(full); }
        return false;
      };

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop();
        for (const line of lines) {
          if (line.startsWith('data:') && handle(line.slice(5).replace(/^ /, ''))) return full;
        }
      }
      if (buffer.startsWith('data:')) handle(buffer.slice(5).replace(/^ /, ''));
      if (!full.trim()) throw new ApiError('The backend stream ended without an answer.');
      return full;
    }
  };

  function describeError(err) {
    if (err instanceof ApiError) return err.message;
    if (!navigator.onLine) return 'You are offline. Reconnect and try again.';
    if (err instanceof TypeError) {
      return 'Could not reach the backend at ' + state.settings.endpoint + '. Check that it is running, that the endpoint in Settings is correct, and that it allows requests from this page (CORS).';
    }
    return err.message || 'Something went wrong. Try again.';
  }

  /* ======================================================================
     7. UI RENDERING
     ====================================================================== */
  const activeConversation = () => state.conversations.find((c) => c.id === state.activeId) || null;

  /* ---- selects and chips ---- */
  function fitSelect(sel) {
    const opt = sel.options[sel.selectedIndex];
    if (!opt) return;
    const cs = getComputedStyle(sel);
    const probe = document.createElement('span');
    probe.style.cssText = 'position:absolute;visibility:hidden;white-space:nowrap;font-family:' + cs.fontFamily +
      ';font-size:' + cs.fontSize + ';font-weight:' + cs.fontWeight + ';letter-spacing:' + cs.letterSpacing;
    probe.textContent = opt.text;
    document.body.appendChild(probe);
    const w = Math.ceil(probe.getBoundingClientRect().width);
    probe.remove();
    sel.style.width = (w + 26) + 'px';
  }

  function fillSelect(sel, items, value) {
    sel.replaceChildren(...items.map((i) => new Option(i.label, i.id)));
    sel.value = items.some((i) => i.id === value) ? value : items[0].id;
  }

  function renderChips() {
    el.subjectGrid.replaceChildren();
    SUBJECTS.filter((s) => s.id !== 'auto').forEach((s) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'subject-chip';
      b.style.setProperty('--h', s.hue);
      b.textContent = s.label;
      b.dataset.subject = s.id;
      b.addEventListener('click', () => {
        setPref('subject', state.prefs.subject === s.id ? 'auto' : s.id);
        if (isDesktop()) el.questionInput.focus();
      });
      el.subjectGrid.appendChild(b);
    });
  }

  function syncPrefsUI() {
    el.subjectSelect.value = state.prefs.subject;
    el.levelSelect.value = state.prefs.level;
    el.modeSelect.value = state.prefs.mode;
    [el.subjectSelect, el.levelSelect, el.modeSelect].forEach(fitSelect);
    el.subjectGrid.querySelectorAll('.subject-chip').forEach((c) => {
      c.setAttribute('aria-pressed', String(c.dataset.subject === state.prefs.subject));
    });
    const hint = (SUBJECTS.find((s) => s.id === state.prefs.subject) || {}).hint;
    el.questionInput.placeholder = hint ? 'e.g. ' + hint : 'Ask a question, or paste an image';
  }

  function setPref(key, value) {
    state.prefs[key] = value;
    Store.set('prefs', state.prefs);
    syncPrefsUI();
  }

  /* ---- composer state ---- */
  function autosize() {
    const ta = el.questionInput;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
  }

  function updateComposerState() {
    const hasInput = el.questionInput.value.trim().length > 0 || state.attachments.length > 0;
    el.sendBtn.disabled = !hasInput;
    el.sendBtn.hidden = state.busy;
    el.stopBtn.hidden = !state.busy;
  }

  function setBusy(flag) {
    state.busy = flag;
    updateComposerState();
  }

  /* ---- scrolling ---- */
  let stick = true;
  el.stage.addEventListener('scroll', () => {
    const gap = el.stage.scrollHeight - el.stage.scrollTop - el.stage.clientHeight;
    stick = gap < 80;
    el.jumpBtn.hidden = gap < 240;
  }, { passive: true });

  function scrollDown(force) {
    if (force) stick = true;
    if (stick) el.stage.scrollTop = el.stage.scrollHeight;
  }

  /* ---- messages ---- */
  function refreshBody(node, msg, { final }) {
    const body = node.querySelector('.msg__body');
    node.classList.toggle('msg--pending', !!msg.pending);
    node.classList.toggle('msg--error', !!msg.error);

    if (msg.role === 'user') {
      body.textContent = msg.content;
      body.hidden = !msg.content;
      return;
    }

    body.className = 'msg__body md';
    body.replaceChildren();

    if (msg.content) {
      body.innerHTML = Markdown.render(msg.content);
      Markdown.enhance(body, final);
    } else if (!msg.error) {
      body.innerHTML = '<span class="typing" role="status" aria-label="Working on your answer"><i></i><i></i><i></i></span>';
      return;
    }
    if (msg.error) {
      const p = document.createElement('p');
      p.className = 'msg__error';
      p.textContent = msg.error;
      body.appendChild(p);
    } else if (msg.stopped) {
      const p = document.createElement('p');
      p.className = 'msg__note';
      p.textContent = 'Stopped before the answer finished.';
      body.appendChild(p);
    }
  }

  function buildMessageEl(msg) {
    const node = tpl('messageTemplate');
    node.dataset.id = msg.id;
    node.classList.add(msg.role === 'user' ? 'msg--user' : 'msg--assistant');

    const avatar = node.querySelector('.msg__avatar');
    const who = node.querySelector('.msg__who');
    if (msg.role === 'assistant') {
      avatar.textContent = '\u03C0';
      who.textContent = APP.name;
    } else {
      who.replaceChildren();
      const m = msg.meta || {};
      [m.subject !== 'auto' ? labelOf(SUBJECTS, m.subject) : '', m.level !== 'auto' ? labelOf(LEVELS, m.level) : '', labelOf(MODES, m.mode)]
        .filter(Boolean)
        .forEach((t) => {
          const s = document.createElement('span');
          s.className = 'tag';
          s.textContent = t;
          who.appendChild(s);
        });
    }

    const box = node.querySelector('.msg__images');
    if (msg.images && msg.images.length) {
      const full = state.sessionImages.get(msg.id) || [];
      msg.images.forEach((im, i) => {
        const img = document.createElement('img');
        img.src = im.thumb;
        img.alt = 'Uploaded image ' + (i + 1);
        img.loading = 'lazy';
        img.dataset.full = (full[i] && full[i].dataUrl) || im.thumb;
        box.appendChild(img);
      });
    } else {
      box.remove();
    }

    refreshBody(node, msg, { final: true });
    return node;
  }

  function markLast() {
    el.messages.querySelectorAll('.msg--last').forEach((n) => n.classList.remove('msg--last'));
    const nodes = el.messages.querySelectorAll('.msg--assistant');
    if (nodes.length) nodes[nodes.length - 1].classList.add('msg--last');
  }

  function renderConversation() {
    const conv = activeConversation();
    el.messages.replaceChildren();
    if (conv) conv.messages.forEach((m) => el.messages.appendChild(buildMessageEl(m)));
    el.welcome.hidden = !!(conv && conv.messages.length);
    const title = conv ? conv.title : 'New conversation';
    el.conversationTitle.textContent = title;
    document.title = conv ? title + ' - ' + APP.name : APP.name;
    markLast();
    scrollDown(true);
    renderHistory();
  }

  /* ---- sidebar history ---- */
  function renderHistory() {
    const q = state.historyFilter.trim().toLowerCase();
    const list = state.conversations
      .filter((c) => !q || c.title.toLowerCase().includes(q) || c.messages.some((m) => (m.content || '').toLowerCase().includes(q)))
      .sort((a, b) => b.updatedAt - a.updatedAt);

    el.conversationList.replaceChildren();
    list.forEach((c) => {
      const node = tpl('conversationItemTemplate');
      node.dataset.id = c.id;
      node.classList.toggle('is-active', c.id === state.activeId);
      node.querySelector('.history__title').textContent = c.title;
      const n = c.messages.length;
      node.querySelector('.history__meta').textContent = n + (n === 1 ? ' message' : ' messages') + ', ' + timeLabel(c.updatedAt);
      node.querySelector('[data-action="open"]').addEventListener('click', () => openConversation(c.id));
      node.querySelector('[data-action="delete"]').addEventListener('click', () => deleteConversation(c.id));
      el.conversationList.appendChild(node);
    });

    el.historyEmpty.hidden = list.length > 0;
    el.historyEmpty.textContent = q ? 'No conversations match your search.' : 'No conversations yet. Ask your first question to start one.';
  }

  /* ======================================================================
     8. CHAT FLOW
     ====================================================================== */
  function persist() {
    const serial = () => state.conversations.map((c) => ({ ...c, messages: c.messages.filter((m) => !m.pending) }));
    let ok = Store.set('conversations', serial());
    let pruned = false;
    while (!ok && state.conversations.length > 1) {
      // Storage full: drop the oldest conversation that is not open.
      const victim = state.conversations
        .filter((c) => c.id !== state.activeId)
        .sort((a, b) => a.updatedAt - b.updatedAt)[0];
      if (!victim) break;
      state.conversations = state.conversations.filter((c) => c !== victim);
      pruned = true;
      ok = Store.set('conversations', serial());
    }
    if (pruned) toast('Storage was full, so the oldest conversation was removed.', 'error');
    Store.set('active', state.activeId);
  }

  function newConversation() {
    state.activeId = null;
    Store.set('active', null);
    renderConversation();
    closeSidebar();
    if (isDesktop()) el.questionInput.focus();
  }

  function openConversation(id) {
    if (!state.conversations.some((c) => c.id === id)) return;
    state.activeId = id;
    Store.set('active', id);
    renderConversation();
    closeSidebar();
  }

  function deleteConversation(id) {
    const conv = state.conversations.find((c) => c.id === id);
    if (!conv || !window.confirm('Delete "' + conv.title + '"? This cannot be undone.')) return;
    state.conversations = state.conversations.filter((c) => c.id !== id);
    conv.messages.forEach((m) => state.sessionImages.delete(m.id));
    if (state.activeId === id) state.activeId = null;
    persist();
    renderConversation();
  }

  function clearAll() {
    if (!state.conversations.length) { toast('There is nothing to clear.'); return; }
    if (!window.confirm('Delete all conversations stored on this device? This cannot be undone.')) return;
    state.conversations = [];
    state.sessionImages.clear();
    state.activeId = null;
    persist();
    renderConversation();
    closeSidebar();
  }

  function titleFrom(text, hasImages) {
    const clean = text.replace(/\s+/g, ' ').trim();
    if (!clean) return hasImages ? 'Image question' : 'New conversation';
    return clean.length > 48 ? clean.slice(0, 47).trimEnd() + '\u2026' : clean;
  }

  function buildHistory(messages) {
    const limit = state.settings.historyLimit;
    if (limit <= 0) return [];
    return messages
      .filter((m) => !m.pending && !m.error && (m.content || (m.images && m.images.length)))
      .map((m) => ({ role: m.role, content: m.content || '[image attached]', has_images: !!(m.images && m.images.length) }))
      .slice(-limit);
  }

  let rafId = 0;
  function scheduleRender(node, msg) {
    if (rafId) return;
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      refreshBody(node, msg, { final: false });
      scrollDown();
    });
  }

  async function handleSubmit() {
    if (state.busy) return;
    const text = el.questionInput.value.trim();
    const images = state.attachments.slice();
    if (!text && !images.length) return;

    let conv = activeConversation();
    if (!conv) {
      conv = { id: uid(), title: titleFrom(text, images.length > 0), createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
      state.conversations.unshift(conv);
      state.activeId = conv.id;
      if (state.conversations.length > LIMITS.maxConversations) {
        state.conversations.sort((a, b) => b.updatedAt - a.updatedAt).length = LIMITS.maxConversations;
      }
    }

    const userMsg = {
      id: uid(),
      role: 'user',
      content: text,
      images: images.map((i) => ({ name: i.name, thumb: i.thumb })),
      ts: Date.now(),
      meta: { subject: state.prefs.subject, level: state.prefs.level, mode: state.prefs.mode }
    };
    if (images.length) state.sessionImages.set(userMsg.id, images);
    conv.messages.push(userMsg);
    conv.updatedAt = Date.now();

    el.questionInput.value = '';
    autosize();
    state.attachments = [];
    renderAttachments();

    el.welcome.hidden = true;
    el.conversationTitle.textContent = conv.title;
    document.title = conv.title + ' - ' + APP.name;
    el.messages.appendChild(buildMessageEl(userMsg));
    persist();
    renderHistory();
    scrollDown(true);

    await runAssistant(conv, userMsg);
  }

  async function runAssistant(conv, userMsg) {
    const index = conv.messages.indexOf(userMsg);
    const history = buildHistory(conv.messages.slice(0, index));
    const images = state.sessionImages.get(userMsg.id) || [];
    if (userMsg.images && userMsg.images.length && !images.length) {
      toast('The original image is no longer in memory, so this request goes without it.');
    }

    const aMsg = { id: uid(), role: 'assistant', content: '', pending: true, ts: Date.now() };
    conv.messages.push(aMsg);
    const node = buildMessageEl(aMsg);
    el.messages.appendChild(node);
    markLast();
    scrollDown(true);

    const controller = new AbortController();
    state.controller = controller;
    setBusy(true);

    const ctx = { conversation: conv, userMessage: userMsg, assistantMessage: aMsg };
    try {
      let payload = AIClient.buildPayload({
        question: userMsg.content || DEFAULT_IMAGE_PROMPT,
        meta: userMsg.meta,
        history,
        images,
        conversationId: conv.id
      });
      for (const hook of PSAI.hooks.beforeSend) payload = (await hook(payload, ctx)) || payload;

      aMsg.content = await AIClient.send(payload, {
        signal: controller.signal,
        onToken: (full) => { aMsg.content = full; scheduleRender(node, aMsg); }
      });

      for (const hook of PSAI.hooks.afterAnswer) {
        try { await hook(aMsg.content, ctx); } catch (e) { console.warn('afterAnswer hook failed', e); }
      }
    } catch (err) {
      if (err && err.name === 'AbortError') {
        aMsg.stopped = true;
        if (!aMsg.content) aMsg.error = 'You stopped this answer before it started. Retry to ask again.';
      } else {
        aMsg.error = describeError(err);
      }
    } finally {
      cancelAnimationFrame(rafId);
      rafId = 0;
      aMsg.pending = false;
      state.controller = null;
      setBusy(false);
      refreshBody(node, aMsg, { final: true });
      conv.updatedAt = Date.now();
      persist();
      renderHistory();
      markLast();
      scrollDown();
    }
  }

  function regenerate() {
    if (state.busy) return;
    const conv = activeConversation();
    if (!conv) return;
    const last = conv.messages[conv.messages.length - 1];
    if (!last || last.role !== 'assistant') return;
    conv.messages.pop();
    const userMsg = [...conv.messages].reverse().find((m) => m.role === 'user');
    const lastNode = el.messages.querySelector('[data-id="' + last.id + '"]');
    if (lastNode) lastNode.remove();
    if (!userMsg) return;
    runAssistant(conv, userMsg);
  }

  /* ======================================================================
     9. SETTINGS, TOOLS, SIDEBAR
     ====================================================================== */
  function openSettings() {
    el.endpointInput.value = state.settings.endpoint;
    el.historyLimitInput.value = state.settings.historyLimit;
    el.imageSizeInput.value = state.settings.imageMaxDim;
    el.streamToggle.checked = state.settings.stream;
    el.settingsDialog.returnValue = '';
    closeSidebar();
    el.settingsDialog.showModal();
  }

  function onSettingsClosed() {
    if (el.settingsDialog.returnValue !== 'save') return;
    const endpoint = el.endpointInput.value.trim() || DEFAULT_SETTINGS.endpoint;
    if (!/^(\/|https?:\/\/)/i.test(endpoint)) {
      toast('The endpoint must start with / or https://', 'error');
      return;
    }
    state.settings.endpoint = endpoint;
    state.settings.historyLimit = clamp(parseInt(el.historyLimitInput.value, 10) || 0, 0, 100);
    state.settings.imageMaxDim = clamp(parseInt(el.imageSizeInput.value, 10) || DEFAULT_SETTINGS.imageMaxDim, 512, 4096);
    state.settings.stream = el.streamToggle.checked;
    Store.set('settings', state.settings);
    toast('Settings saved');
  }

  function renderTools() {
    fillSelect(el.modelSelect, MODELS, state.prefs.model);
    el.toolList.replaceChildren(...FEATURES.map((f) => {
      const li = tpl('toolItemTemplate');
      li.querySelector('.tool__name').textContent = f.name;
      li.querySelector('.tool__desc').textContent = f.desc;
      li.querySelector('.tool__status').textContent = f.available ? 'Available' : 'Coming soon';
      const sw = li.querySelector('.tool__switch');
      sw.setAttribute('aria-label', f.name);
      sw.disabled = !f.available;
      sw.checked = !!f.available && !!state.settings.features[f.id];
      sw.addEventListener('change', () => {
        state.settings.features[f.id] = sw.checked;
        Store.set('settings', state.settings);
      });
      return li;
    }));
  }

  const desktopQuery = window.matchMedia('(min-width: 900px)');
  function syncSidebarInert() {
    el.sidebar.inert = !(desktopQuery.matches || el.app.dataset.sidebar === 'open');
  }
  function openSidebar() {
    el.app.dataset.sidebar = 'open';
    el.sidebarScrim.hidden = false;
    el.sidebarOpenBtn.setAttribute('aria-expanded', 'true');
    syncSidebarInert();
  }
  function closeSidebar() {
    delete el.app.dataset.sidebar;
    el.sidebarScrim.hidden = true;
    el.sidebarOpenBtn.setAttribute('aria-expanded', 'false');
    syncSidebarInert();
  }

  /* ======================================================================
     10. EVENTS, PUBLIC API, STARTUP
     ====================================================================== */
  const PSAI = {
    version: APP.version,
    // Async functions that may modify the request: (payload, ctx) => payload
    // Use them later for screen capture, PDFs, memory, SymPy verification, web research.
    hooks: { beforeSend: [], afterAnswer: [] },
    registerModel(model) {
      if (!model || !model.id) return;
      const i = MODELS.findIndex((m) => m.id === model.id);
      if (i >= 0) MODELS[i] = { ...MODELS[i], ...model }; else MODELS.push({ label: model.id, ...model });
      renderTools();
    },
    registerFeature(feature) {
      if (!feature || !feature.id) return;
      const i = FEATURES.findIndex((f) => f.id === feature.id);
      if (i >= 0) FEATURES[i] = { ...FEATURES[i], ...feature }; else FEATURES.push({ name: feature.id, desc: '', available: false, ...feature });
      renderTools();
    },
    client: AIClient
  };
  window.PersonalSuperAI = PSAI;

  function bindEvents() {
    // Composer
    el.composerForm.addEventListener('submit', (e) => { e.preventDefault(); handleSubmit(); });
    el.questionInput.addEventListener('input', () => { autosize(); updateComposerState(); });
    el.questionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        handleSubmit();
      }
    });
    el.questionInput.addEventListener('paste', (e) => {
      const files = Array.from((e.clipboardData && e.clipboardData.files) || []).filter((f) => f.type.startsWith('image/'));
      if (files.length) { e.preventDefault(); addFiles(files); }
    });
    el.stopBtn.addEventListener('click', () => { if (state.controller) state.controller.abort(); });

    // Image upload
    el.attachBtn.addEventListener('click', () => el.fileInput.click());
    el.cameraBtn.addEventListener('click', () => el.cameraInput.click());
    [el.fileInput, el.cameraInput].forEach((input) => {
      input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });
    });
    document.addEventListener('dragover', (e) => { if (e.dataTransfer && e.dataTransfer.types.includes('Files')) e.preventDefault(); });
    document.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files.length) { e.preventDefault(); addFiles(e.dataTransfer.files); }
    });

    // Selectors
    el.subjectSelect.addEventListener('change', () => setPref('subject', el.subjectSelect.value));
    el.levelSelect.addEventListener('change', () => setPref('level', el.levelSelect.value));
    el.modeSelect.addEventListener('change', () => setPref('mode', el.modeSelect.value));

    // Messages: copy, retry, code copy, image zoom
    el.messages.addEventListener('click', async (e) => {
      const codeBtn = e.target.closest('.code-copy');
      if (codeBtn) {
        const pre = codeBtn.closest('pre');
        const code = pre.querySelector('code');
        const ok = await copyText((code || pre).textContent);
        codeBtn.textContent = ok ? 'Copied' : 'Failed';
        setTimeout(() => { codeBtn.textContent = 'Copy'; }, 1500);
        return;
      }
      const img = e.target.closest('.msg__images img');
      if (img) { openLightbox(img.dataset.full || img.src); return; }

      const btn = e.target.closest('[data-action]');
      if (!btn) return;
      if (btn.dataset.action === 'regenerate') { regenerate(); return; }
      if (btn.dataset.action === 'copy') {
        const conv = activeConversation();
        const id = btn.closest('.msg').dataset.id;
        const msg = conv && conv.messages.find((m) => m.id === id);
        if (!msg || !msg.content) return;
        const ok = await copyText(msg.content);
        const label = btn.querySelector('span');
        const use = btn.querySelector('use');
        btn.classList.toggle('is-done', ok);
        label.textContent = ok ? 'Copied' : 'Copy failed';
        if (ok) use.setAttribute('href', '#i-check');
        setTimeout(() => {
          btn.classList.remove('is-done');
          label.textContent = 'Copy';
          use.setAttribute('href', '#i-copy');
        }, 1600);
      }
    });

    el.jumpBtn.addEventListener('click', () => scrollDown(true));

    // Sidebar
    el.sidebarOpenBtn.addEventListener('click', openSidebar);
    el.sidebarCloseBtn.addEventListener('click', closeSidebar);
    el.sidebarScrim.addEventListener('click', closeSidebar);
    el.newChatBtn.addEventListener('click', newConversation);
    el.newChatTopBtn.addEventListener('click', newConversation);
    el.clearAllBtn.addEventListener('click', clearAll);
    el.historySearch.addEventListener('input', () => { state.historyFilter = el.historySearch.value; renderHistory(); });
    desktopQuery.addEventListener('change', () => { if (desktopQuery.matches) closeSidebar(); syncSidebarInert(); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && el.app.dataset.sidebar === 'open') closeSidebar();
    });

    // Dialogs
    el.settingsBtn.addEventListener('click', openSettings);
    el.settingsDialog.addEventListener('close', onSettingsClosed);
    el.settingsResetBtn.addEventListener('click', () => {
      el.endpointInput.value = DEFAULT_SETTINGS.endpoint;
      el.historyLimitInput.value = DEFAULT_SETTINGS.historyLimit;
      el.imageSizeInput.value = DEFAULT_SETTINGS.imageMaxDim;
      el.streamToggle.checked = DEFAULT_SETTINGS.stream;
    });
    el.toolsBtn.addEventListener('click', () => { renderTools(); el.toolsDialog.showModal(); });
    el.modelSelect.addEventListener('change', () => setPref('model', el.modelSelect.value));
    [el.settingsDialog, el.toolsDialog, el.lightbox].forEach((d) => {
      d.addEventListener('click', (e) => { if (e.target === d) d.close(); });
    });
    el.lightbox.addEventListener('close', () => { el.lightboxImg.removeAttribute('src'); });
  }

  function loadState() {
    state.settings = { ...DEFAULT_SETTINGS, ...Store.get('settings', {}) };
    state.settings.features = state.settings.features || {};
    state.prefs = { ...state.prefs, ...Store.get('prefs', {}) };

    const saved = Store.get('conversations', []);
    state.conversations = (Array.isArray(saved) ? saved : [])
      .filter((c) => c && c.id && Array.isArray(c.messages))
      .map((c) => {
        c.messages = c.messages.filter((m) => !m.pending);
        const last = c.messages[c.messages.length - 1];
        if (last && last.role === 'user') {
          // The page was closed before an answer arrived. Offer a retry.
          c.messages.push({ id: uid(), role: 'assistant', content: '', error: 'This answer did not finish. Retry to ask again.', ts: Date.now() });
        }
        c.title = c.title || 'New conversation';
        c.updatedAt = c.updatedAt || c.createdAt || Date.now();
        return c;
      });

    const active = Store.get('active', null);
    state.activeId = state.conversations.some((c) => c.id === active) ? active : null;
  }

  function init() {
    loadState();
    fillSelect(el.subjectSelect, SUBJECTS, state.prefs.subject);
    fillSelect(el.levelSelect, LEVELS, state.prefs.level);
    fillSelect(el.modeSelect, MODES, state.prefs.mode);
    renderChips();
    renderTools();
    syncPrefsUI();
    bindEvents();
    renderAttachments();
    renderConversation();
    updateComposerState();
    syncSidebarInert();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => syncPrefsUI());
  }

  init();
})();
