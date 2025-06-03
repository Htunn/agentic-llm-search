# FOFA Integration

This document explains how to use the FOFA integration features in the Agentic LLM Search application.

## Overview

[FOFA](https://fofa.info/) is a search engine for Internet-connected devices and services, similar to Shodan. Our integration allows you to:

1. Search for devices and services using FOFA's search syntax
2. Get detailed information about specific IP addresses
3. Access statistics about search results
4. View FOFA account information
5. Use the LLM agent to analyze and explain FOFA results

## Setup

1. **Get FOFA API Credentials**: Sign up at [FOFA.info](https://fofa.info/toLogin) to get an account.

2. **Add Your API Credentials**: Update your `.env` file to include your FOFA email and API key:
   ```
   FOFA_EMAIL=your_registered_email@example.com
   FOFA_API_KEY=your_fofa_api_key_here
   ```

## Using the Command Line Interface

The application includes a dedicated command-line interface for FOFA searches:

```bash
# Search for devices
python fofa_cli.py search "domain=example.com" --limit 10

# Get detailed information about an IP address
python fofa_cli.py host 8.8.8.8

# Get statistics for a query
python fofa_cli.py stats "domain=example.com"

# Get API account information
python fofa_cli.py info

# Check your F-points balance
python fofa_cli.py points
```

### Advanced Usage

Use the `--agent` flag to have the LLM analyze the results:

```bash
# Use the LLM agent to analyze search results
python fofa_cli.py search "domain=example.com" --agent

# Use the LLM agent to analyze host information
python fofa_cli.py host 8.8.8.8 --agent
```

Use the `--raw` flag to get JSON output:

```bash
# Get raw JSON output
python fofa_cli.py search "domain=example.com" --raw
```

Specify custom fields to include in the results:

```bash
# Get specific fields in the results
python fofa_cli.py search "domain=example.com" --fields "ip,port,server,title"
```

## Using the Web Interface

1. Start the Streamlit web interface:
   ```bash
   streamlit run app.py
   ```

2. In the sidebar, ensure "Enable FOFA Search" is checked.

3. Change the "Search Type" to "FOFA" and select either "Query" or "Host Lookup".

4. Enter your search query or IP address and click "Ask".

### Example FOFA Queries

- Basic search: `domain=example.com`
- Search by country: `country="US" && server="Apache"`
- Search by port: `port=22 && protocol=ssh`
- Search by product: `title="nginx"`
- Complex search: `domain="example.com" && country="US" && port=443`

## Programmatic Usage

You can use the FOFA integration in your Python code:

```python
from src.agents.agentic_llm import AgenticLLMAgent

# Initialize the agent
agent = AgenticLLMAgent()

# Asynchronous search for devices
response = await agent.process_fofa_query("domain=example.com", limit=5)

# Asynchronous host lookup
host_response = await agent.process_fofa_host("8.8.8.8")

# Access results
for source in response.sources:
    print(f"{source.title}: {source.url}")
    print(source.content[:100] + "...")
```

For direct access to the FOFA API without the agent:

```python
from src.tools.fofa_tool import FofaSearchTool

# Initialize the FOFA tool
fofa = FofaSearchTool()  # Uses credentials from .env file

# Search for devices
results = fofa.search("domain=example.com", limit=10)

# Get host information
host_info = fofa.host_info("8.8.8.8")

# Get statistics
stats = fofa.stats("domain=example.com")

# Get account information
account = fofa.account_info()
```

## Testing the Integration

To verify that your FOFA integration is working correctly:

```bash
# Make the test script executable
chmod +x test_fofa.py

# Run the tests
./test_fofa.py
```

## FOFA Search Syntax

FOFA uses a specific query syntax. Here are some examples:

- Search by domain: `domain="example.com"`
- Search by IP: `ip="8.8.8.8/24"`
- Search by port: `port=443`
- Search by server: `server="Apache"`
- Search by title: `title="Admin Login"`
- Search by country: `country="US"`
- Search by city: `city="San Francisco"`
- Combine searches: `domain="example.com" && port=443 && country="US"`

For more information on FOFA search syntax, visit the [FOFA Documentation](https://fofa.info/help).

## Understanding FOFA F-Points

FOFA uses a credit system called "F-points" for API usage:

- Each API request consumes F-points based on the operation
- Free accounts have limited points that replenish over time
- Search operations consume points based on:
  - The number of results requested
  - The complexity of the query
  - The fields included in the results
- If you receive an error like `[820031] F点余额不足`, it means your account has insufficient F-points
- You can check your F-points balance with `python fofa_cli.py points`
- To increase your limit, consider upgrading your FOFA account at [FOFA VIP](https://fofa.info/static_pages/vip)

## Security Considerations

- FOFA is a powerful tool that reveals information about internet-facing devices
- Use responsibly and ethically
- Do not use this tool to access systems without permission
- Always respect privacy and legal restrictions
- Be aware that API usage may be rate-limited based on your FOFA account level
