// Pokemon Showdown Coach - Content Script
// Reads battle state and displays AI move recommendations

const API_URL = 'http://localhost:5000/recommend';
let lastBattleState = null;
let coachPanel = null;

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

// Extract battle state from Pokemon Showdown's JavaScript
function extractBattleState() {
    // Pokemon Showdown stores battle data in app.curRoom.battle
    if (typeof app === 'undefined' || !app.curRoom || !app.curRoom.battle) {
        return null;
    }

    const battle = app.curRoom.battle;

    // Check if it's our turn
    if (!battle.myPokemon || battle.ended) {
        return null;
    }

    try {
        const state = {
            // Our team
            myTeam: battle.myPokemon.map(p => ({
                species: p.speciesForme || p.species,
                hp: p.hp,
                maxHp: p.maxhp,
                status: p.status || null,
                active: p.active || false,
                fainted: p.fainted || false,
                moves: p.moves || [],
                ability: p.ability || p.baseAbility,
                item: p.item
            })),

            // Our active Pokemon
            myActive: null,

            // Opponent's visible Pokemon
            opponentTeam: [],
            opponentActive: null,

            // Field conditions
            weather: battle.weather || null,
            terrain: battle.terrain || null,

            // Available actions
            availableMoves: [],
            availableSwitches: [],

            // Turn info
            turn: battle.turn,

            // Is it our turn to move?
            waitingForChoice: battle.request && !battle.request.wait
        };

        // Get active Pokemon details
        if (battle.mySide && battle.mySide.active && battle.mySide.active[0]) {
            const active = battle.mySide.active[0];
            state.myActive = {
                species: active.speciesForme || active.species,
                hp: active.hp,
                maxHp: active.maxhp,
                status: active.status,
                boosts: active.boosts || {},
                volatiles: Object.keys(active.volatiles || {})
            };
        }

        // Get opponent's active Pokemon
        if (battle.farSide && battle.farSide.active && battle.farSide.active[0]) {
            const opp = battle.farSide.active[0];
            state.opponentActive = {
                species: opp.speciesForme || opp.species,
                hp: opp.hp,
                maxHp: opp.maxhp,
                status: opp.status,
                boosts: opp.boosts || {},
                volatiles: Object.keys(opp.volatiles || {})
            };
        }

        // Get opponent's revealed Pokemon
        if (battle.farSide && battle.farSide.pokemon) {
            state.opponentTeam = battle.farSide.pokemon.map(p => ({
                species: p.speciesForme || p.species,
                hp: p.hp,
                maxHp: p.maxhp,
                status: p.status,
                fainted: p.fainted,
                revealed: true
            }));
        }

        // Get available moves from the request
        if (battle.request && battle.request.active && battle.request.active[0]) {
            const activeReq = battle.request.active[0];
            if (activeReq.moves) {
                state.availableMoves = activeReq.moves.map((m, i) => ({
                    id: m.id,
                    name: m.move,
                    pp: m.pp,
                    maxPp: m.maxpp,
                    disabled: m.disabled || false,
                    index: i
                }));
            }
        }

        // Get available switches
        if (battle.request && battle.request.side && battle.request.side.pokemon) {
            state.availableSwitches = battle.request.side.pokemon
                .filter((p, i) => i > 0 && !p.fainted && p.hp > 0)
                .map((p, i) => ({
                    species: p.speciesForme || p.species,
                    hp: p.hp,
                    maxHp: p.maxhp,
                    index: i + 1
                }));
        }

        return state;
    } catch (e) {
        console.error('[PS Coach] Error extracting battle state:', e);
        return null;
    }
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
        status.textContent = 'Coach offline - start the server';
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

// Main loop - check for battle state changes
function pollBattleState() {
    const state = extractBattleState();

    if (state && state.waitingForChoice) {
        // We're in a battle and it's our turn
        const stateStr = JSON.stringify(state);

        if (stateStr !== lastBattleState) {
            lastBattleState = stateStr;

            // Show panel if hidden
            if (!coachPanel) createCoachPanel();
            coachPanel.style.display = 'block';

            // Get recommendations
            getRecommendations(state).then(updatePanel);
        }
    } else if (!state) {
        // Not in a battle
        lastBattleState = null;
        if (coachPanel) {
            coachPanel.querySelector('.coach-status').textContent = 'Waiting for battle...';
            coachPanel.querySelector('.coach-recommendations').innerHTML = '';
        }
    }
}

// Initialize
console.log('[PS Coach] Pokemon Showdown Coach loaded');
createCoachPanel();

// Poll every 500ms
setInterval(pollBattleState, 500);
