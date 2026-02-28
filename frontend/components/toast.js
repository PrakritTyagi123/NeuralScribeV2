/** Toast notification. */
let container = null;

function ensureContainer() {
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    return container;
}

export function showToast(message, duration = 3000) {
    const c = ensureContainer();
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    c.appendChild(toast);
    setTimeout(() => {
        toast.remove();
    }, duration);
}