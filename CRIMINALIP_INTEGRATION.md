# Criminal IP Integration for Agentic LLM Search

This document explains how the Criminal IP cybersecurity search engine is integrated into the agentic-llm-search project.

## What is Criminal IP?

Criminal IP is a cybersecurity search engine that provides information about IP addresses, domains, and internet-connected assets. It helps security researchers identify threats, vulnerabilities, and malicious activities on the internet.

Key features of Criminal IP include:
- IP reputation and security scoring
- Domain analysis and security assessment
- Banner grabbing and service identification
- Malicious activity detection
- Asset discovery and monitoring

## Integration Components

The Criminal IP integration consists of the following components:

1. **CriminalIPTool** (`src/tools/criminalip_tool.py`) - Core component that interfaces with the Criminal IP API, providing methods for searching and formatting results.

2. **InternetSearchTool Integration** (`src/tools/search_tool.py`) - Extends the existing search tool with Criminal IP capabilities.

3. **CLI Tool** (`criminalip_cli.py`) - Standalone command-line interface for interacting with Criminal IP.

4. **Agent Integration** - Extends the AgenticLLMAgent to support Criminal IP queries.

## Using Criminal IP Search

### API Key Setup

To use Criminal IP features, you need a Criminal IP API key:

1. Create an account at [https://www.criminalip.io/](https://www.criminalip.io/)
2. Generate an API key in your account dashboard

You have two options for setting up your API key:

### Option 1: Using the CLI setup command (Recommended)

Use the included CLI tool to set up your API key:

```bash
# Set up with interactive prompt
./criminalip_cli.py setup

# Or provide the key directly
./criminalip_cli.py setup --api-key YOUR_API_KEY
```

This will validate your API key and save it to your `.env` file.

### Option 2: Manual Setup

Add the API key to your `.env` file manually:

```
CRIMINAL_IP_API_KEY=your_api_key_here
```

### Using the CLI Tool

The `criminalip_cli.py` script provides command-line access to Criminal IP features:

```bash
# Get information about an IP address
python criminalip_cli.py ip 8.8.8.8

# Search for assets matching a query
python criminalip_cli.py search "apache 2.4"

# Get information about a domain
python criminalip_cli.py domain example.com

# Search for banner information
python criminalip_cli.py banner "nginx"

# View API account information
python criminalip_cli.py info
```

### Using in the Agent

The Criminal IP search is integrated with the main agentic LLM search engine:

```python
from src.agents.agentic_llm import AgenticLLM

# Initialize the agent
agent = AgenticLLM()

# Perform a Criminal IP search
response = agent.search_and_respond("What security issues does IP 8.8.8.8 have?")

# Process a specific IP
response = agent.process_criminalip_host("8.8.8.8")
```

## Available Methods

The Criminal IP tool provides the following methods:

- `search_ip(ip_address)` - Get security information about an IP address
- `get_ip_summary(ip_address)` - Get summary information about an IP address
- `check_ip_malicious(ip_address)` - Check if an IP is associated with malicious activity
- `search_domain(domain, limit)` - Search for domain information
- `asset_search(query, limit)` - Search for assets matching a query
- `banner_search(query, limit)` - Search for banner information
- `get_user_info()` - Get account information

## Data Types

Criminal IP searches return information about the following entities:

### IP Address Data

- Security score (0-100)
- Open ports and services
- Hosting/server information
- Geolocation data
- Malicious activity indicators
- ISP/ASN information

### Domain Data

- Security score
- Associated IP addresses
- DNS records (A, MX, NS, etc.)
- SSL certificate information
- WHOIS data

### Banner/Asset Data

- Service identification
- Software versions
- Configuration details
- Vulnerability indicators

## Comparison with Other Tools

| Feature           | Criminal IP              | Shodan                   | FOFA                     |
|-------------------|--------------------------|--------------------------| ------------------------ |
| Focus             | Security & threat intel  | IoT & service discovery  | Service & cert discovery |
| Strength          | Malicious IP detection   | Broad device coverage    | Flexible query syntax    |
| API Limit         | Daily/monthly limits     | Credits-based            | Points-based             |
| Integration       | Full                     | Full                     | Full                     |
