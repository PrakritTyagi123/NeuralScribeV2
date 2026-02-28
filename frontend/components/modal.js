/** Modal dialog. */
export function showModal(title, contentEl) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.remove();
    });

    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `<div class="modal-title">${title}</div>`;

    const body = document.createElement('div');
    if (typeof contentEl === 'string') {
        body.innerHTML = contentEl;
    } else {
        body.appendChild(contentEl);
    }
    modal.appendChild(body);

    const closeBtn = document.createElement('button');
    closeBtn.className = 'btn mt-16';
    closeBtn.textContent = 'Close';
    closeBtn.addEventListener('click', () => overlay.remove());
    modal.appendChild(closeBtn);

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    return { close: () => overlay.remove() };
}