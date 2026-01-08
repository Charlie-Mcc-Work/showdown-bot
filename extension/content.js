// Pokemon Showdown Coach - Content Script
// Handles UI and API communication (runs in isolated world)

const API_URL = 'http://localhost:5000/recommend';
let coachPanel = null;
let lastStateJson = null;

// Create the coach panel UI
function createCoachPanel() {
    if (coachPanel) return coachPanel;

    coachPanel = document.createElement('div');
    coachPanel.id = 'ps-coach-panel';
    coachPanel.innerHTML = `
        <div class="coach-header">
            <span class="coach-title">AI Coach</span>
            <button class="coach-toggle" title="Minimize">_</button>
        </div>
        <div class="coach-content">
            <div class="coach-status">Waiting for battle...</div>
            <div class="coach-recommendations"></div>
        </div>
    `;
    document.body.appendChild(coachPanel);

    // Toggle minimize
    coachPanel.querySelector('.coach-toggle').addEventListener('click', () => {
        coachPanel.classList.toggle('minimized');
    });

    return coachPanel;
}

// Send state to API and get recommendations
async function getRecommendations(state) {
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state)
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return await response.json();
    } catch (e) {
        console.error('[PS Coach] API error:', e);
        return { error: e.message };
    }
}

// Update the coach panel with recommendations
function updatePanel(recommendations) {
    if (!coachPanel) return;

    const content = coachPanel.querySelector('.coach-recommendations');
    const status = coachPanel.querySelector('.coach-status');

    if (recommendations.error) {
        status.textContent = 'Coach offline - start server';
        status.className = 'coach-status error';
        content.innerHTML = `
            <div class="coach-help">
                Run: <code>python scripts/coach_server.py</code>
            </div>
        `;
        return;
    }

    if (!recommendations.moves || recommendations.moves.length === 0) {
        status.textContent = 'Analyzing...';
        status.className = 'coach-status';
        return;
    }

    status.textContent = 'Recommendations:';
    status.className = 'coach-status active';

    content.innerHTML = recommendations.moves.map((move, i) => `
        <div class="coach-move ${i === 0 ? 'best' : ''}">
            <span class="move-rank">${i + 1}</span>
            <span class="move-name">${move.name}</span>
            <span class="move-score">${(move.score * 100).toFixed(0)}%</span>
        </div>
    `).join('');
}

// Handle messages from the page script (runs in MAIN world)
window.addEventListener('message', async (event) => {
    if (event.source !== window) return;
    if (!event.data || event.data.type !== 'PS_COACH_BATTLE_STATE') return;

    const state = event.data.state;
    const status = coachPanel?.querySelector('.coach-status');

    if (!state) {
        // Not in battle or not our turn
        if (status) {
            if (event.data.reason === 'no_battle') {
                status.textContent = 'Waiting for battle...';
            } else if (event.data.reason === 'not_our_turn') {
                status.textContent = "Opponent's turn...";
            } else {
                status.textContent = 'Waiting...';
            }
            status.className = 'coach-status';
        }
        // Clear recommendations when not our turn
        if (coachPanel) {
            coachPanel.querySelector('.coach-recommendations').innerHTML = '';
        }
        lastStateJson = null;
        return;
    }

    // Only update if state changed
    const stateJson = JSON.stringify(state);
    if (stateJson === lastStateJson) return;
    lastStateJson = stateJson;

    // Get recommendations from API
    const recommendations = await getRecommendations(state);
    updatePanel(recommendations);
});

// Initialize
console.log('[PS Coach] Content script loaded');
createCoachPanel();
