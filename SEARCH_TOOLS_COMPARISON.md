# Search Tools Comparison: Criminal IP vs FOFA vs Shodan

This document provides a comparison between the Criminal IP, FOFA, and Shodan integrations in the Agentic LLM Search application.

## Overview

Criminal IP, FOFA, and Shodan are search engines for Internet-connected devices and cybersecurity intelligence, but they have different features, coverage, and pricing models. This comparison will help you choose the appropriate tool for your specific needs.

## Feature Comparison

| Feature | Criminal IP | FOFA | Shodan |
|---------|-------------|------|--------|
| **Search Focus** | Security intelligence and threat detection | Web assets and infrastructure | All internet-connected devices |
| **Specialization** | IP reputation and malicious activity detection | Service identification and attributes | Device discovery and categorization |
| **Query Syntax** | Parameter-based API with specific endpoints | Uses `field="value"` with `&&` operators | Uses `field:value` with space-separated terms |
| **Coverage** | Global coverage with threat intelligence | Strong in Asian regions, especially China | Global coverage with strong US/EU presence |
| **Credit System** | Daily/monthly API request limits | F-points system with consumption per query | Credit system with query/scan credits |
| **Data Refresh** | Near real-time updates for security data | Regular updates | Regular updates |
| **API Structure** | REST API with specialized endpoints | REST API | REST API |
| **Historical Data** | Security history and past incidents | Limited historical data | Extensive historical data (paid plans) |
| **Pricing Model** | Free tier with daily limits, paid tiers | Free tier with F-points limitation | Free tier with query limitations |
| **Security Scoring** | Comprehensive scoring system | Basic categorization | Basic categorization |

## Integration Implementation

Our application implements all three tools with similar architecture patterns:

1. Core Tool Class (`CriminalIPTool` / `FofaSearchTool` / `ShodanSearchTool`)
2. Integration with the `InternetSearchTool` class
3. LLM Agent methods for analysis 
4. Command Line Interface for each tool
5. Web Interface integration

## When to Use Which Tool

### Use Criminal IP when:

- You need detailed security information about IP addresses
- You're investigating potential malicious activity
- You want security scores and reputation information
- You need comprehensive domain security analysis
- You're performing threat intelligence research

### Use FOFA when:

- You need stronger coverage of Asian/Chinese infrastructure
- You prefer the FOFA query syntax
- You have FOFA credits or subscription
- You need specific metadata that FOFA provides
- You're primarily interested in service identification

### Use Shodan when:

- You need global device discovery
- You prefer Shodan's query syntax
- You want to use Shodan's faceting capabilities
- You have Shodan credits or subscription
- You need historical data (paid plans)

## Common Use Cases

### Security Research

Both tools are useful for security research, such as:
- Finding exposed devices
- Identifying vulnerable systems
- Conducting internet-wide surveys

### Infrastructure Monitoring

Monitor your organization's internet exposure:
- Check for unauthorized services
- Verify security policies
- Monitor cloud assets

### Competitive Analysis

- Research competitor infrastructure
- Identify technology stacks
- Understand deployment patterns

## Conclusion

Having Criminal IP, FOFA, and Shodan integrations provides complementary capabilities for cybersecurity research. Each tool has different strengths, and your specific use case will determine which is most appropriate. Our application makes it easy to switch between them or use multiple tools for comprehensive results.

For detailed usage instructions, refer to:
- [Criminal IP Integration Documentation](CRIMINALIP_INTEGRATION.md)
- [FOFA Integration Documentation](FOFA_INTEGRATION.md)
- [Shodan Integration Documentation](SHODAN_INTEGRATION.md)
