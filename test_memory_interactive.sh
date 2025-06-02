#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Conversation Memory Testing Script ===${NC}"
echo -e "This script will help you interactively test the conversation memory feature."
echo

echo -e "${BLUE}Running the agent in interactive mode with memory enabled...${NC}"
echo -e "${YELLOW}Try the following test sequence:${NC}"
echo "1. Ask a question about a specific topic (e.g., 'What is machine learning?')"
echo "2. Ask a follow-up question using pronouns (e.g., 'What are its applications?')"
echo "3. Use /memory off to disable memory"
echo "4. Ask another follow-up question - note how it doesn't use previous context"
echo "5. Use /memory on to enable memory again"
echo "6. Use /clear to reset the conversation"
echo "7. Start a new topic"
echo
echo -e "${BLUE}The /config command will show you the memory status${NC}"

echo
echo -e "${GREEN}Starting the agent...${NC}"
echo

# Run the main.py script in interactive mode
python3 main.py
