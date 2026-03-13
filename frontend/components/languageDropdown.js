/**
 * NeuralScribe v2 — Language Dropdown Component
 * Reusable dropdown that fetches languages from the API and fires a callback on change.
 *
 * Usage:
 *   const dropdown = await createLanguageDropdown(async (language) => { ... });
 *   container.appendChild(dropdown.element);
 *   dropdown.getValue();  // current language
 */

/**
 * Create a language dropdown.
 * @param {Function} onChange - async callback(language) when selection changes
 * @param {string} [context] - which UI state key to use (prep_language, training_language, inference_language)
 * @returns {Promise<{element: HTMLElement, getValue: Function, setValue: Function, refresh: Function}>}
 */
export async function createLanguageDropdown(onChange, context = 'prep_language') {
    const wrapper = document.createElement('div');
    wrapper.className = 'lang-dropdown';

    const label = document.createElement('label');
    label.className = 'lang-dropdown-label';
    label.textContent = 'Language';
    wrapper.appendChild(label);

    const select = document.createElement('select');
    select.className = 'lang-dropdown-select';
    wrapper.appendChild(select);

    const status = document.createElement('span');
    status.className = 'lang-dropdown-status';
    wrapper.appendChild(status);

    let currentLanguage = 'english';

    async function loadLanguages() {
        try {
            const res = await fetch('/api/system/languages');
            const data = await res.json();
            const languages = data.languages || [];
            currentLanguage = data.selected || 'english';

            select.innerHTML = '';
            languages.forEach(lang => {
                const opt = document.createElement('option');
                opt.value = lang.id;
                opt.textContent = lang.name;
                if (!lang.enabled) {
                    opt.textContent += ' (coming soon)';
                    opt.disabled = true;
                }
                if (lang.id === currentLanguage) {
                    opt.selected = true;
                }
                select.appendChild(opt);
            });

            updateStatus(languages.find(l => l.id === currentLanguage));
        } catch (e) {
            select.innerHTML = '<option value="english">English</option>';
            status.textContent = '';
        }
    }

    function updateStatus(langInfo) {
        if (!langInfo) {
            status.textContent = '';
            return;
        }
        if (langInfo.dataset_prepared) {
            status.textContent = '✓ Dataset ready';
            status.className = 'lang-dropdown-status ready';
        } else if (langInfo.status === 'placeholder') {
            status.textContent = '○ Not available';
            status.className = 'lang-dropdown-status placeholder';
        } else {
            status.textContent = '○ No dataset';
            status.className = 'lang-dropdown-status pending';
        }
    }

    select.addEventListener('change', async () => {
        const newLang = select.value;
        if (newLang === currentLanguage) return;
        currentLanguage = newLang;

        // Update status display
        status.textContent = 'Switching...';
        status.className = 'lang-dropdown-status switching';

        try {
            if (onChange) await onChange(newLang);
            // Refresh to get updated status
            await loadLanguages();
        } catch (e) {
            status.textContent = 'Error';
            status.className = 'lang-dropdown-status error';
        }
    });

    await loadLanguages();

    return {
        element: wrapper,
        getValue() { return currentLanguage; },
        setValue(lang) {
            select.value = lang;
            currentLanguage = lang;
        },
        async refresh() { await loadLanguages(); },
    };
}

/**
 * Create a model selection dropdown for the current language.
 * @param {Function} onChange - async callback(modelName) when selection changes
 * @param {string} language - current language
 * @returns {Promise<{element: HTMLElement, getValue: Function, refresh: Function}>}
 */
export async function createModelDropdown(onChange, language = 'english') {
    const wrapper = document.createElement('div');
    wrapper.className = 'model-dropdown';

    const label = document.createElement('label');
    label.className = 'lang-dropdown-label';
    label.textContent = 'Model';
    wrapper.appendChild(label);

    const select = document.createElement('select');
    select.className = 'lang-dropdown-select';
    wrapper.appendChild(select);

    const info = document.createElement('span');
    info.className = 'lang-dropdown-status';
    wrapper.appendChild(info);

    let currentModel = null;

    async function loadModels(lang) {
        language = lang || language;
        try {
            const res = await fetch(`/api/models/list?language=${language}`);
            const data = await res.json();
            const models = data.models || [];

            select.innerHTML = '';

            if (models.length === 0) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'No models — train first';
                opt.disabled = true;
                opt.selected = true;
                select.appendChild(opt);
                info.textContent = '';
                currentModel = null;
                return;
            }

            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.name;
                opt.textContent = m.name + (m.is_best ? ' ★' : '') +
                    (m.val_acc != null ? ` (${(m.val_acc * 100).toFixed(1)}%)` : '');
                select.appendChild(opt);
            });

            // Select best model by default
            const best = models.find(m => m.is_best);
            if (best) {
                select.value = best.name;
                currentModel = best.name;
            } else {
                currentModel = models[0].name;
            }

            // Check what's loaded
            const statusRes = await fetch('/api/inference/status');
            const statusData = await statusRes.json();
            if (statusData.loaded_model) {
                select.value = statusData.loaded_model;
                currentModel = statusData.loaded_model;
                info.textContent = '✓ Loaded';
                info.className = 'lang-dropdown-status ready';
            } else {
                info.textContent = '○ Not loaded';
                info.className = 'lang-dropdown-status pending';
            }
        } catch (e) {
            select.innerHTML = '<option value="">Error loading models</option>';
            info.textContent = '';
        }
    }

    select.addEventListener('change', async () => {
        const name = select.value;
        if (!name || name === currentModel) return;
        currentModel = name;
        info.textContent = 'Loading...';
        info.className = 'lang-dropdown-status switching';

        try {
            if (onChange) await onChange(name);
            info.textContent = '✓ Loaded';
            info.className = 'lang-dropdown-status ready';
        } catch (e) {
            info.textContent = 'Error';
            info.className = 'lang-dropdown-status error';
        }
    });

    await loadModels(language);

    return {
        element: wrapper,
        getValue() { return currentModel; },
        async refresh(lang) { await loadModels(lang); },
    };
}