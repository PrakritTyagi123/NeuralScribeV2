/**
 * NeuralScribe v2 — Simple SPA router.
 * Maps view names to render functions. Handles sidebar navigation.
 */

import { renderDashboard } from '../views/dashboard.js';
import { renderDataPrep } from '../views/dataPrep.js';
import { renderTraining } from '../views/training.js';
import { renderModelManager } from '../views/modelManager.js';
import { renderInference } from '../views/inference.js';
import { renderExplainability } from '../views/explainability.js';
import { renderSetting } from '../views/setting.js';

const views = {
    dashboard: renderDashboard,
    dataPrep: renderDataPrep,
    training: renderTraining,
    modelManager: renderModelManager,
    inference: renderInference,
    explainability: renderExplainability,
    setting: renderSetting,
};

let currentView = 'dashboard';

export function initRouter() {
    // Sidebar click handlers
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            if (view) navigateTo(view);
        });
    });

    // Render initial view
    navigateTo('dashboard');
}

export function navigateTo(viewName) {
    if (!views[viewName]) return;

    currentView = viewName;

    // Update sidebar active state
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });

    // Render view
    const main = document.getElementById('main');
    main.innerHTML = '';
    main.className = '';  // Reset any view-specific classes
    views[viewName](main);
}

export function getCurrentView() {
    return currentView;
}