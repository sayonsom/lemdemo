#!/bin/bash
# Script to test the LEM API endpoints

# Configuration
API_URL="http://localhost:8000"

# Check if jq is installed
if command -v jq &> /dev/null; then
    JQ_AVAILABLE=true
else
    JQ_AVAILABLE=false
    echo "Note: jq is not installed. JSON responses will not be pretty-printed."
    echo "To install jq on macOS: brew install jq"
    echo "To install jq on Ubuntu: sudo apt-get install jq"
    echo ""
fi

# Function to format JSON response
format_json() {
    if [ "$JQ_AVAILABLE" = true ]; then
        jq .
    else
        cat
    fi
}

# Test health endpoint
echo "Testing health endpoint..."
curl -s $API_URL/health
echo -e "\n"

# Test preprocess endpoint
echo "Testing preprocess endpoint..."
curl -s -X POST $API_URL/preprocess \
  -H "Content-Type: application/json" \
  -d '{"device": "ac_unit", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}' | format_json
echo -e "\n"

# Test embed endpoint
echo "Testing embed endpoint..."
curl -s -X POST $API_URL/embed \
  -H "Content-Type: application/json" \
  -d '[{"device": "light", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}, {"device": "door", "capability": "lock", "attributes": {"state": "UNLOCKED"}, "timestamp": "2023-01-01T08:01:00"}]' | format_json
echo -e "\n"

# Test suggest endpoint
echo "Testing suggest endpoint..."
curl -s -X POST $API_URL/suggest \
  -H "Content-Type: application/json" \
  -d '[{"device": "light", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}, {"device": "door", "capability": "lock", "attributes": {"state": "UNLOCKED"}, "timestamp": "2023-01-01T08:01:00"}]' | format_json
echo -e "\n"

echo "API testing complete!"
echo "You can also access the interactive API documentation at: $API_URL/docs" 