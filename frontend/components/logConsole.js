/** Log console component. */
export function createLogConsole() {
    const el = document.createElement('div');
    el.className = 'log-console';
    el.id = 'log-console';
    return el;
}

export function appendLog(container, message, level = 'info') {
    const el = container.querySelector ? container : document.getElementById('log-console');
    if (!el) return;
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;
    el.appendChild(entry);
    el.scrollTop = el.scrollHeight;

    // Keep max 200 entries
    while (el.children.length > 200) el.removeChild(el.firstChild);
}