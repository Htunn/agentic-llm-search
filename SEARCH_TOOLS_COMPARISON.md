# Search Tools Comparison: FOFA vs Shodan

This document provides a comparison between the FOFA and Shodan integrations in the Agentic LLM Search application.

## Overview

Both FOFA and Shodan are search engines for Internet-connected devices, but they have different features, coverage, and pricing models. This comparison will help you choose the appropriate tool for your needs.

## Feature Comparison

| Feature | FOFA | Shodan |
|---------|------|--------|
| **Search Focus** | More focus on web assets and infrastructure | Broader focus on all internet-connected devices |
| **Query Syntax** | Uses `field="value"` with `&&` operators | Uses `field:value` with space-separated terms |
| **Coverage** | Strong in Asian regions, especially China | Global coverage with strong US/EU presence |
| **Credit System** | F-points system with consumption per query | Credit system with query/scan credits |
| **Data Refresh** | Regular updates | Regular updates |
| **API Structure** | REST API | REST API |
| **Historical Data** | Limited historical data | Extensive historical data (paid plans) |
| **Pricing Model** | Free tier with F-points limitation | Free tier with query limitations |

## Integration Implementation

Our application implements both tools with similar architecture patterns:

1. Core Tool Class (`FofaSearchTool` / `ShodanSearchTool`)
2. Integration with the `InternetSearchTool` class
3. LLM Agent methods for analysis 
4. Command Line Interface
5. Web Interface integration

## When to Use Which Tool

### Use FOFA when:

- You need stronger coverage of Asian/Chinese infrastructure
- You prefer the FOFA query syntax
- You have FOFA credits or subscription
- You need specific metadata that FOFA provides

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

Having both FOFA and Shodan integrations provides complementary capabilities. The tools have different strengths, and your specific use case will determine which is more appropriate. Our application makes it easy to switch between them or use both for comprehensive results.

For detailed usage instructions, refer to:
- [FOFA Integration Documentation](FOFA_INTEGRATION.md)
- [Shodan Integration Documentation](SHODAN_INTEGRATION.md)
