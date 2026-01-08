#!/bin/bash
# Start all services needed for local browser play against the bot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CLIENT_DIR="$HOME/pokemon-showdown-client"
SERVER_DIR="$HOME/pokemon-showdown"
BROWSER_DIR="$PROJECT_DIR/browser"

echo "=============================================="
echo "Pokemon Showdown - Local Play Setup"
echo "=============================================="

# Check if Pokemon Showdown server is running
if ! pgrep -f "pokemon-showdown" > /dev/null; then
    echo "Starting Pokemon Showdown server..."
    cd "$SERVER_DIR"
    node pokemon-showdown start --no-security &
    sleep 3
else
    echo "Pokemon Showdown server already running"
fi

# Check if HTTP server for client is running on port 8080
if ! ss -tlnp | grep -q ":8080 "; then
    echo "Starting client HTTP server on port 8080..."
    cd "$CLIENT_DIR"
    npx http-server -p 8080 -c-1 &
    sleep 2
else
    echo "Client HTTP server already running on 8080"
fi

# Serve the setup page on port 8081
if ! ss -tlnp | grep -q ":8081 "; then
    echo "Starting setup page server on port 8081..."
    cd "$BROWSER_DIR"
    python3 -m http.server 8081 &
    sleep 1
else
    echo "Setup page server already running on 8081"
fi

echo ""
echo "=============================================="
echo "Services are running!"
echo "=============================================="
echo ""
echo "QUICK START - Open this page for setup instructions:"
echo "   http://localhost:8081/local-client.html"
echo ""
echo "OR manually:"
echo ""
echo "1. Start the bot in another terminal:"
echo "   cd $PROJECT_DIR"
echo "   .venv-rocm/bin/python scripts/play.py"
echo ""
echo "2. Open testclient:"
echo "   http://localhost:8080/play.pokemonshowdown.com/testclient.html?~~localhost:8000"
echo ""
echo "3. Login via console (F12): app.socket.send('|/trn YourName,0,')"
echo ""
echo "4. Challenge 'TrainedBot'"
echo ""
echo "Press Ctrl+C to stop background services"
echo "=============================================="

# Wait for interrupt
wait
