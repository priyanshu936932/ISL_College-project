/* ============================================================
   ISL Recognition — Enhancements
   All features are purely additive — nothing in this file
   modifies or replaces existing app.js logic.
   ============================================================ */

(function () {
  'use strict';

  /* ── wait for DOM ──────────────────────────────────────── */
  function ready(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn);
    } else {
      fn();
    }
  }

  /* ── sign icons (mirror of app.js mapping) ─────────────── */
  const ICONS = {
    alive: '❤️', bad: '👎', female: '👩', good: '👍',
    happy: '😊', long: '🙌', male: '👨', default: '👋'
  };

  /* ══════════════════════════════════════════════════════════
     1. SERVER STATUS INDICATOR
     ══════════════════════════════════════════════════════════ */
  function initServerStatus() {
    const logo = document.querySelector('.nav-logo');
    if (!logo) return;

    const dot = document.createElement('span');
    dot.className = 'server-status-dot checking';
    dot.setAttribute('data-tip', 'Checking backend…');
    logo.appendChild(dot);

    async function ping() {
      dot.className = 'server-status-dot checking';
      dot.setAttribute('data-tip', 'Checking backend…');
      try {
        const res = await fetch('http://localhost:5000/health', { signal: AbortSignal.timeout(4000) });
        if (res.ok) {
          dot.className = 'server-status-dot online';
          dot.setAttribute('data-tip', '✅ Backend Online');
        } else {
          throw new Error('not ok');
        }
      } catch {
        dot.className = 'server-status-dot offline';
        dot.setAttribute('data-tip', '❌ Backend Offline');
      }
    }

    ping();
    setInterval(ping, 30_000);
  }

  /* ══════════════════════════════════════════════════════════
     2. ANIMATED STAT COUNTERS
     ══════════════════════════════════════════════════════════ */
  function initStatCounters() {
    const statNumbers = document.querySelectorAll('.stat-number');
    if (!statNumbers.length) return;

    /* parse "63M+" → { value: 63, suffix: "M+" } */
    function parse(text) {
      const match = text.trim().match(/^([\d.]+)(.*)$/);
      if (!match) return null;
      return { value: parseFloat(match[1]), suffix: match[2] };
    }

    /* easing: ease-out cubic */
    function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

    function animateCounter(el, target, suffix, duration) {
      const start = performance.now();
      function step(now) {
        const t = Math.min((now - start) / duration, 1);
        const current = Math.round(easeOut(t) * target);
        el.textContent = current + suffix;
        if (t < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    /* store originals and zero-out */
    const targets = [];
    statNumbers.forEach(el => {
      const parsed = parse(el.textContent);
      if (parsed) {
        targets.push({ el, ...parsed });
        el.setAttribute('data-target', parsed.value);
      }
    });

    const observer = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting)) {
        observer.disconnect();
        targets.forEach(({ el, value, suffix }) => {
          animateCounter(el, value, suffix, 1800);
        });
      }
    }, { threshold: 0.4 });

    const container = document.querySelector('.stat-counters');
    if (container) observer.observe(container);
  }

  /* ══════════════════════════════════════════════════════════
     3. ANALYZING SPINNER
     ══════════════════════════════════════════════════════════ */
  function initAnalyzingSpinner() {
    const predictionName = document.getElementById('prediction-name');
    const predictionCard = document.querySelector('.prediction-card');
    if (!predictionName || !predictionCard) return;

    const mo = new MutationObserver(() => {
      const text = predictionName.textContent.trim();
      if (text === 'Processing...') {
        predictionCard.classList.add('analyzing');
      } else {
        predictionCard.classList.remove('analyzing');
      }
    });

    mo.observe(predictionName, { childList: true, characterData: true, subtree: true });
  }

  /* ══════════════════════════════════════════════════════════
     4. PREDICTION HISTORY PANEL
     ══════════════════════════════════════════════════════════ */
  function initPredictionHistory() {
    const uploadSection = document.querySelector('.upload-section');
    if (!uploadSection) return;

    /* inject panel after upload section */
    const panel = document.createElement('div');
    panel.id = 'pred-history';
    panel.innerHTML = `
      <div class="pred-history-inner">
        <div class="pred-history-header">
          <h3>📋 Prediction History</h3>
          <button class="pred-history-clear" id="pred-history-clear">🗑 Clear</button>
        </div>
        <div id="pred-history-body"></div>
      </div>
    `;
    uploadSection.insertAdjacentElement('afterend', panel);

    const body = document.getElementById('pred-history-body');
    const clearBtn = document.getElementById('pred-history-clear');

    const STORAGE_KEY = 'isl_pred_history';
    const MAX_ITEMS = 5;

    function load() {
      try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
      catch { return []; }
    }

    function save(list) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
    }

    function relativeTime(ts) {
      const diff = Math.floor((Date.now() - ts) / 1000);
      if (diff < 60) return 'just now';
      if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
      if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`;
      return `${Math.floor(diff / 86400)} day ago`;
    }

    function render() {
      const list = load();
      if (!list.length) {
        body.innerHTML = '<p class="pred-history-empty">No predictions yet — analyze a video to get started.</p>';
        return;
      }
      body.innerHTML = `<div class="pred-history-list">
        ${list.map(item => `
          <div class="pred-history-item">
            <div class="pred-history-emoji">${ICONS[item.gesture] || ICONS.default}</div>
            <div class="pred-history-info">
              <div class="pred-history-name">${item.gesture.toUpperCase()}</div>
              <div class="pred-history-conf-track">
                <div class="pred-history-conf-fill" style="width:${(item.confidence * 100).toFixed(1)}%"></div>
              </div>
            </div>
            <div class="pred-history-time">${(item.confidence * 100).toFixed(1)}%<br>${relativeTime(item.ts)}</div>
          </div>`).join('')}
      </div>`;
    }

    function push(gesture, confidence) {
      const list = load();
      list.unshift({ gesture, confidence, ts: Date.now() });
      save(list.slice(0, MAX_ITEMS));
      render();
    }

    clearBtn.addEventListener('click', () => {
      save([]);
      render();
    });

    render();

    /* watch prediction-name for real results */
    const predName = document.getElementById('prediction-name');
    const predIcon = document.getElementById('prediction-icon');
    const confValue = document.getElementById('confidence-value');
    if (!predName) return;

    const SKIP = new Set(['Ready to Predict', 'Processing...', 'Error', '']);

    let lastSeen = '';
    const mo = new MutationObserver(() => {
      const text = predName.textContent.trim();
      if (SKIP.has(text) || text === lastSeen) return;
      lastSeen = text;

      /* parse confidence from #confidence-value e.g. "97.4%" */
      const rawConf = confValue ? confValue.textContent.replace('%', '').trim() : '0';
      const confidence = parseFloat(rawConf) / 100 || 0;

      push(text.toLowerCase(), confidence);
    });

    mo.observe(predName, { childList: true, characterData: true, subtree: true });
  }

  /* ══════════════════════════════════════════════════════════
     5. COPY RESULT BUTTON
     ══════════════════════════════════════════════════════════ */
  function initCopyButton() {
    const predCard = document.querySelector('.prediction-card');
    const predName = document.getElementById('prediction-name');
    const confValue = document.getElementById('confidence-value');
    if (!predCard || !predName) return;

    const btn = document.createElement('button');
    btn.id = 'copy-result-btn';
    btn.textContent = '📋 Copy Result';
    predCard.appendChild(btn);

    btn.addEventListener('click', () => {
      const gesture = predName.textContent.trim();
      const conf = confValue ? confValue.textContent.trim() : '';
      const text = `ISL Gesture: ${gesture} — ${conf} confidence`;
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = '✅ Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = '📋 Copy Result';
          btn.classList.remove('copied');
        }, 2000);
      }).catch(() => {});
    });

    const SKIP = new Set(['Ready to Predict', 'Processing...', 'Error', '']);
    const mo = new MutationObserver(() => {
      const text = predName.textContent.trim();
      btn.style.display = SKIP.has(text) ? 'none' : 'block';
    });
    mo.observe(predName, { childList: true, characterData: true, subtree: true });
  }

  /* ══════════════════════════════════════════════════════════
     6. PARTICLE CANVAS
     ══════════════════════════════════════════════════════════ */
  function initParticles() {
    const hero = document.querySelector('.hero');
    if (!hero) return;

    const canvas = document.createElement('canvas');
    canvas.id = 'particle-canvas';
    hero.insertBefore(canvas, hero.firstChild);

    const ctx = canvas.getContext('2d');
    let W, H, particles, raf;

    const COUNT = 60;
    const COLORS = ['#7c3aed', '#a78bfa', '#14b8a6', '#ec4899', '#f1f5f9'];

    function resize() {
      W = canvas.width  = hero.offsetWidth;
      H = canvas.height = hero.offsetHeight;
    }

    function randomParticle() {
      return {
        x:    Math.random() * W,
        y:    Math.random() * H,
        r:    Math.random() * 2 + 0.8,
        vx:   (Math.random() - 0.5) * 0.5,
        vy:   (Math.random() - 0.5) * 0.5,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
        alpha: Math.random() * 0.5 + 0.2,
      };
    }

    function init() {
      resize();
      particles = Array.from({ length: COUNT }, randomParticle);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = W;
        if (p.x > W) p.x = 0;
        if (p.y < 0) p.y = H;
        if (p.y > H) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function loop() {
      draw();
      raf = requestAnimationFrame(loop);
    }

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        cancelAnimationFrame(raf);
      } else {
        loop();
      }
    });

    window.addEventListener('resize', () => { resize(); });

    init();
    loop();
  }

  /* ── boot all features ─────────────────────────────────── */
  ready(() => {
    initServerStatus();
    initStatCounters();
    initAnalyzingSpinner();
    initPredictionHistory();
    initCopyButton();
    initParticles();
  });

})();
