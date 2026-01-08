// Pokemon Showdown Coach - Page Script
// Runs in MAIN world to access Pokemon Showdown's app object

(function() {
    'use strict';

    function extractBattleState() {
        if (typeof app === 'undefined' || !app.curRoom || !app.curRoom.battle) {
            return { state: null, reason: 'no_battle' };
        }

        const battle = app.curRoom.battle;
        const request = app.curRoom.request;  // Request is on the room, not battle

        if (!battle.myPokemon || battle.ended) {
            return { state: null, reason: 'no_battle' };
        }

        // Check if we have a request (it's our turn)
        if (!request || request.wait) {
            return { state: null, reason: 'not_our_turn' };
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
                    fainted: p.fainted || (p.hp === 0),
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

                // It's our turn
                waitingForChoice: true
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
            if (request && request.active && request.active[0]) {
                const activeReq = request.active[0];
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
            if (request && request.side && request.side.pokemon) {
                state.availableSwitches = request.side.pokemon
                    .map((p, i) => ({ ...p, originalIndex: i }))
                    .filter((p, i) => i > 0 && !p.fainted && p.condition !== '0 fnt')
                    .map((p) => ({
                        species: p.speciesForme || p.details.split(',')[0],
                        hp: parseInt(p.condition.split('/')[0]) || 0,
                        maxHp: parseInt(p.condition.split('/')[1]) || 100,
                        index: p.originalIndex
                    }));
            }

            return { state: state, reason: null };
        } catch (e) {
            console.error('[PS Coach] Error extracting battle state:', e);
            return { state: null, reason: 'error' };
        }
    }

    function pollAndSend() {
        const result = extractBattleState();
        window.postMessage({
            type: 'PS_COACH_BATTLE_STATE',
            state: result.state,
            reason: result.reason
        }, '*');
    }

    // Poll every 500ms
    setInterval(pollAndSend, 500);

    // Initial poll
    setTimeout(pollAndSend, 100);

    console.log('[PS Coach] Page script loaded (MAIN world)');
})();
