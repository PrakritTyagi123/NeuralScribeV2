/**
 * NeuralScribe v2 — SPA Router
 * Pages: Intro → Download → Data Prep → Training → Models → Live View → Settings
 */
import { renderIntro } from '../views/intro.js';
import { renderDownload } from '../views/download.js';
import { renderDataPrep } from '../views/dataPrep.js';
import { renderTraining } from '../views/training.js';
import { renderModelManager } from '../views/modelManager.js';
import { renderExplainability } from '../views/explainability.js';
import { renderSetting } from '../views/setting.js';

const views = {
    intro: renderIntro,
    download: renderDownload,
    dataPrep: renderDataPrep,
    training: renderTraining,
    modelManager: renderModelManager,
    explainability: renderExplainability,
    setting: renderSetting,
};

let currentView = 'intro';

export async function initRouter() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            if (view) navigateTo(view);
        });
    });
    navigateTo('intro');
}

export function navigateTo(viewName) {
    if (!views[viewName]) return;
    currentView = viewName;
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });
    const main = document.getElementById('main');
    main.innerHTML = '';
    main.className = '';
    views[viewName](main);
}

export function getCurrentView() { return currentView; }
