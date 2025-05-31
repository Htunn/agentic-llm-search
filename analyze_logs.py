#!/usr/bin/env python3
"""
Log Analyzer Tool for Agentic LLM Search

This script analyzes log files from the Agentic LLM Search system and identifies common errors and issues.
It provides recommendations for fixing problems based on log patterns.
"""

import os
import re
import sys
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    print("Note: 'rich' module not found. Install with 'pip install rich' for better formatting.")


class LogAnalyzer:
    """Analyzer for LLM log files"""
    
    def __init__(self):
        """Initialize the log analyzer with common error patterns"""
        self.error_patterns = {
            'gpu_error': r'(CUDA|MPS|GPU).*(error|not available|failed|problem)',
            'token_limit': r'(token|context).*(limit|exceeded|overflow)',
            'model_not_found': r'(model|file).*(not found|missing|doesn\'t exist)',
            'memory_error': r'(memory|RAM).*(error|insufficient|not enough)',
            'timeout_error': r'(timeout|timed out)',
            'import_error': r'(import|module).*not found',
            'openai_api_error': r'(OpenAI|API).*(error|key|invalid|rate limit)',
            'hf_transfer_error': r'(hf_transfer|huggingface).*(error|failed)',
        }
        
        # Common solutions for identified issues
        self.solutions = {
            'gpu_error': [
                "Check if GPU is properly configured with 'python check_gpu.py'",
                "For Apple Silicon, ensure USE_METAL=True in .env file",
                "Reduce GPU_LAYERS in .env file (try value between 16-32)",
                "Fall back to CPU with USE_GPU=False in .env file"
            ],
            'token_limit': [
                "Increase CONTEXT_LENGTH in .env (try 4096)",
                "Reduce the length of input prompts",
                "Break down long requests into shorter segments"
            ],
            'model_not_found': [
                "Run 'python download_model.py' to download the TinyLlama model",
                "Check if model path in .env is correct",
                "Ensure model file exists in src/models directory"
            ],
            'memory_error': [
                "Try a smaller model variant",
                "Reduce batch size or context length",
                "Close other memory-intensive applications",
                "Increase virtual memory/swap space"
            ],
            'timeout_error': [
                "Check your internet connection",
                "Increase timeout values in config file",
                "Try again later (external service might be temporarily unavailable)"
            ],
            'import_error': [
                "Run 'pip install -r requirements.txt' to install dependencies",
                "Create a new virtual environment with 'python -m venv venv'",
                "Check for compatibility issues between packages"
            ],
            'openai_api_error': [
                "Check if OPENAI_API_KEY is correctly set in .env file",
                "Verify API key validity in OpenAI dashboard",
                "Check for rate limiting or usage quotas"
            ],
            'hf_transfer_error': [
                "Run 'python install_hf_transfer.py' to fix HuggingFace transfer issues",
                "Check network connectivity to HuggingFace servers",
                "Try manual download from HuggingFace website"
            ]
        }
    
    def find_log_files(self, directory=None):
        """Find log files in the specified directory or current directory"""
        if directory is None:
            directory = os.getcwd()
        
        log_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        
        # Also check common log directories
        log_dirs = ['logs', 'log', 'var/log']
        for log_dir in log_dirs:
            path = os.path.join(directory, log_dir)
            if os.path.exists(path) and os.path.isdir(path):
                for file in os.listdir(path):
                    if file.endswith('.log'):
                        log_files.append(os.path.join(path, file))
        
        return log_files
    
    def analyze_log(self, log_file):
        """Analyze a log file for common errors and issues"""
        if not os.path.exists(log_file):
            print(f"Log file not found: {log_file}")
            return None
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading log file: {e}")
            return None
        
        results = {
            'file': log_file,
            'size': os.path.getsize(log_file),
            'modified': datetime.fromtimestamp(os.path.getmtime(log_file)),
            'issues': [],
            'error_types': Counter(),
            'total_errors': 0
        }
        
        for error_type, pattern in self.error_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            match_count = 0
            for match in matches:
                match_count += 1
                context_start = max(0, match.start() - 100)
                context_end = min(len(content), match.end() + 100)
                issue_context = content[context_start:context_end].strip()
                
                # Try to extract the specific line with the error
                lines = issue_context.split('\n')
                error_line = None
                for line in lines:
                    if match.group(0) in line:
                        error_line = line
                        break
                
                results['issues'].append({
                    'type': error_type,
                    'line': error_line or issue_context.split('\n')[0],
                    'context': issue_context
                })
            
            results['error_types'][error_type] = match_count
            results['total_errors'] += match_count
        
        # Check for general exceptions
        exception_matches = re.finditer(r'(Exception|Error).*:', content)
        for match in exception_matches:
            context_start = max(0, match.start() - 100)
            context_end = min(len(content), match.end() + 100)
            issue_context = content[context_start:context_end].strip()
            
            # Extract the specific error line
            error_line = None
            lines = issue_context.split('\n')
            for line in lines:
                if match.group(0) in line:
                    error_line = line
                    break
            
            results['issues'].append({
                'type': 'general_exception',
                'line': error_line or issue_context.split('\n')[0],
                'context': issue_context
            })
            results['error_types']['general_exception'] = results['error_types'].get('general_exception', 0) + 1
            results['total_errors'] += 1
        
        return results
    
    def get_recommendations(self, issue_type):
        """Get recommendations for fixing an issue type"""
        return self.solutions.get(issue_type, [
            "Check the installation and setup",
            "Verify dependency versions in requirements.txt",
            "Run diagnostics.py to identify system issues"
        ])
    
    def print_analysis(self, results):
        """Print the analysis results"""
        if not results:
            return
        
        # Print file info
        if HAS_RICH:
            console.print(f"[bold blue]Analysis for:[/bold blue] {results['file']}")
            console.print(f"[bold blue]Size:[/bold blue] {results['size'] / 1024:.2f} KB")
            console.print(f"[bold blue]Modified:[/bold blue] {results['modified']}")
            console.print(f"[bold blue]Total issues found:[/bold blue] {results['total_errors']}")
            console.print("")
        else:
            print(f"Analysis for: {results['file']}")
            print(f"Size: {results['size'] / 1024:.2f} KB")
            print(f"Modified: {results['modified']}")
            print(f"Total issues found: {results['total_errors']}")
            print("")
        
        # Print issue summary
        if results['total_errors'] > 0:
            if HAS_RICH:
                table = Table(title="Issue Summary")
                table.add_column("Issue Type", style="cyan")
                table.add_column("Count", justify="right", style="green")
                
                for issue_type, count in results['error_types'].items():
                    table.add_row(issue_type.replace('_', ' ').title(), str(count))
                
                console.print(table)
                console.print("")
            else:
                print("Issue Summary:")
                print("-------------")
                for issue_type, count in results['error_types'].items():
                    print(f"{issue_type.replace('_', ' ').title()}: {count}")
                print("")
            
            # Print detailed issues
            for i, issue in enumerate(results['issues'][:10], 1):  # Limit to 10 issues
                if HAS_RICH:
                    panel = Panel(
                        f"[red]{issue['line'].strip()}[/red]\n\n" +
                        f"[yellow]Context:[/yellow] {issue['context'][:100]}...",
                        title=f"Issue #{i} - {issue['type'].replace('_', ' ').title()}",
                        border_style="red"
                    )
                    console.print(panel)
                    
                    # Print recommendations
                    recommendations = self.get_recommendations(issue['type'])
                    console.print("[bold blue]Recommendations:[/bold blue]")
                    for j, rec in enumerate(recommendations, 1):
                        console.print(f"  [green]{j}.[/green] {rec}")
                    console.print("")
                else:
                    print(f"Issue #{i} - {issue['type'].replace('_', ' ').title()}")
                    print("-" * 50)
                    print(f"Line: {issue['line'].strip()}")
                    print(f"Context: {issue['context'][:100]}...")
                    print("")
                    
                    # Print recommendations
                    recommendations = self.get_recommendations(issue['type'])
                    print("Recommendations:")
                    for j, rec in enumerate(recommendations, 1):
                        print(f"  {j}. {rec}")
                    print("")
            
            if len(results['issues']) > 10:
                if HAS_RICH:
                    console.print(f"[yellow]... and {len(results['issues']) - 10} more issues (showing first 10 only)[/yellow]")
                else:
                    print(f"... and {len(results['issues']) - 10} more issues (showing first 10 only)")
        else:
            if HAS_RICH:
                console.print("[bold green]No issues found in this log file![/bold green]")
            else:
                print("No issues found in this log file!")


def main():
    """Main function for the log analyzer tool"""
    parser = argparse.ArgumentParser(description='Analyze log files for the Agentic LLM Search system')
    parser.add_argument('--log', '-l', help='Path to the log file to analyze')
    parser.add_argument('--dir', '-d', help='Directory to search for log files')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all log files found in the directory')
    args = parser.parse_args()
    
    analyzer = LogAnalyzer()
    
    if args.log:
        # Analyze specific log file
        results = analyzer.analyze_log(args.log)
        if results:
            analyzer.print_analysis(results)
    elif args.dir or args.all:
        # Find and analyze logs in directory
        directory = args.dir if args.dir else os.getcwd()
        log_files = analyzer.find_log_files(directory)
        
        if not log_files:
            if HAS_RICH:
                console.print("[bold yellow]No log files found in the specified directory.[/bold yellow]")
            else:
                print("No log files found in the specified directory.")
            return
        
        if HAS_RICH:
            console.print(f"[bold blue]Found {len(log_files)} log file(s)[/bold blue]")
        else:
            print(f"Found {len(log_files)} log file(s)")
        
        for log_file in log_files:
            if HAS_RICH:
                console.print(f"\n[bold]{'='*60}[/bold]")
            else:
                print(f"\n{'='*60}")
            results = analyzer.analyze_log(log_file)
            if results:
                analyzer.print_analysis(results)
    else:
        # No specific file, try to find most recent logs
        log_files = analyzer.find_log_files()
        
        if not log_files:
            if HAS_RICH:
                console.print("[bold yellow]No log files found in the current directory structure.[/bold yellow]")
                console.print("[bold yellow]Specify a log file with --log or a directory with --dir[/bold yellow]")
            else:
                print("No log files found in the current directory structure.")
                print("Specify a log file with --log or a directory with --dir")
            return
        
        # Sort by modification time, most recent first
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        most_recent = log_files[0]
        
        if HAS_RICH:
            console.print(f"[bold blue]Analyzing most recent log file: {most_recent}[/bold blue]")
        else:
            print(f"Analyzing most recent log file: {most_recent}")
            
        results = analyzer.analyze_log(most_recent)
        if results:
            analyzer.print_analysis(results)


if __name__ == "__main__":
    main()
