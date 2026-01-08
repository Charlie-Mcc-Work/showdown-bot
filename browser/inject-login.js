// Pokemon Showdown Local Login Bypass
// This script auto-logs in to a local Pokemon Showdown server
// without requiring external authentication.
//
// Usage: Paste this into your browser console after loading the testclient,
// or use a userscript manager like Tampermonkey.

(function() {
    'use strict';

    // Configuration
    const DEFAULT_USERNAME = 'Player';
    const AUTO_LOGIN_DELAY = 1000; // ms after page load

    // Wait for the app to be ready
    function waitForApp(callback) {
        if (typeof app !== 'undefined' && app.socket && app.socket.readyState === 1) {
            callback();
        } else {
            setTimeout(() => waitForApp(callback), 100);
        }
    }

    // Perform local login
    function localLogin(username) {
        username = username || DEFAULT_USERNAME;

        // The key insight: poke-env uses "/trn username,0," to login
        // The format is: /trn username,avatar_num,assertion
        // With --no-security, the server accepts empty assertion
        if (typeof app !== 'undefined' && app.socket) {
            console.log('[Local Login] Logging in as: ' + username);
            app.socket.send('|/trn ' + username + ',0,');

            // Update the UI
            if (typeof app.user !== 'undefined') {
                app.user.rename(username);
            }

            console.log('[Local Login] Login command sent!');
            console.log('[Local Login] You can now challenge other users (e.g., TrainedBot)');
        } else {
            console.error('[Local Login] App not ready. Try again in a moment.');
        }
    }

    // Patch the login system to use local auth
    function patchLoginSystem() {
        // Override the login form submission
        if (typeof app !== 'undefined' && app.user) {
            const originalRename = app.user.rename;
            app.user.rename = function(name, token) {
                if (!token || token === '') {
                    // Local login - send /trn directly
                    console.log('[Local Login] Using local auth for: ' + name);
                    app.socket.send('|/trn ' + name + ',0,');
                    return;
                }
                // Fall back to original for authenticated logins
                return originalRename.apply(this, arguments);
            };
            console.log('[Local Login] Login system patched for local server');
        }
    }

    // Auto-login on page load
    function autoLogin() {
        waitForApp(function() {
            patchLoginSystem();

            // Check if we're on a local server
            const isLocal = window.location.href.includes('localhost') ||
                           window.location.href.includes('127.0.0.1');

            if (isLocal) {
                const urlParams = new URLSearchParams(window.location.search);
                const username = urlParams.get('username') || DEFAULT_USERNAME;

                // Show login prompt
                const doLogin = confirm(
                    'Pokemon Showdown Local Server\n\n' +
                    'Click OK to login as "' + username + '"\n' +
                    'Click Cancel to choose a different name'
                );

                if (doLogin) {
                    localLogin(username);
                } else {
                    const newName = prompt('Enter your username:', username);
                    if (newName) {
                        localLogin(newName);
                    }
                }
            }
        });
    }

    // Expose function globally for manual use
    window.localLogin = localLogin;

    // Run auto-login
    setTimeout(autoLogin, AUTO_LOGIN_DELAY);

    console.log('[Local Login] Script loaded. Use localLogin("YourName") to login manually.');
})();
