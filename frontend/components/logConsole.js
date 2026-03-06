// Simple in-memory buffer so logs persist across tab changes
// and are never culled while the app is running.
const _logBuffer = [];

/** Log console component. */
export function createLogConsole() {
    const el = document.createElement('div');
    el.className = 'log-console';
    el.id = 'log-console';

    // Replay any existing buffered entries so logs survive
    // when switching views.
    _logBuffer.forEach(html => {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = html;
        el.appendChild(entry);
    });
    el.scrollTop = el.scrollHeight;

    return el;
}

export function appendLog(container, message, level = 'info') {
    const el = container.querySelector ? container : document.getElementById('log-console');
    const time = new Date().toLocaleTimeString();
    const html = `<span class="log-time">[${time}]</span> ${message}`;

    // Always record into buffer first so it's not lost when
    // the view is unmounted.  Cap at 2000 entries.
    _logBuffer.push(html);
    if (_logBuffer.length > 2000) _logBuffer.splice(0, _logBuffer.length - 2000);

    if (!el) return;
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = html;
    el.appendChild(entry);
    el.scrollTop = el.scrollHeight;
}