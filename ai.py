import os
import sys
import json
import argparse
import datetime
import requests
import platform
import logging
from contextlib import nullcontext
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
import config  # Edit the config.py file

console = Console()

DEBUG = False
HISTORY_FILE = '.aicli_history.json'
LAST_OUTPUT_FILE = '.aicli_last'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    'base': ["You are a helpful AI assistant."],
    'security_audit': ["Your task is to perform a thorough security audit on the provided code, identifying potential vulnerabilities and suggesting mitigations."],
    'extract_wisdom': ["Your task is to extract wisdom, summarize, and provide key insights from the content. Do not leave out any important facts or details that may influence the understanding. Organizing information in bullet points is preffered."],
    'explain_code': ["Your task is to explain the provided code in detail, breaking down its functionality and purpose. Do not explain the basic fundamentals or simple functions, such as printing."],
    'optimize_code': ["Your task is to analyze the provided code and suggest optimizations for improved performance or readability. Repeat the code in full as provided, but with brief comments on the same line."],
    'find_bugs': ["Your task is to carefully analyze the code for potential bugs, issues, or vulnerabilities."],
    'document': ["Your task is to generate comprehensive documentation for the provided code, including function descriptions and usage examples."],
    'architect': ["Your task is to propose a detailed software architecture for the described problem, considering scalability and maintainability."],
    'refactor': ["Your task is to suggest a comprehensive refactoring strategy for the provided code, improving its structure and maintainability."],
    'news': ["Your task is summarize the provided news feed. Seperate the news in segments, like International News, National News, Science and Tech, Cyber Security, and so on. Do not leave out any important details. Ensure you summarize each item. Do not include advertisements."]
}

PROMPT_MAP = {
    'security_audit': "Perform a security audit on the provided code",
    'extract_wisdom': "Summarize and extract key insights",
    'explain_code': "Explain the provided code",
    'optimize_code': "Suggest optimizations",
    'find_bugs': "Analyze code for potential issues",
    'document': "Generate documentation for code",
    'architect': "Propose architecture for the described problem",
    'refactor': "Suggest refactoring strategys for the provided code"
}

class ColorHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        kwargs['max_help_position'] = 30
        super().__init__(*args, **kwargs)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        colored = [f"\033[32m{opt}\033[0m" for opt in action.option_strings]
        return ', '.join(colored)

def get_os():
    """Get the operating system information."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/etc/os-release") as f:
                lines = f.readlines()
            os_info = dict(line.strip().split("=", 1) for line in lines if "=" in line)
            return "{0} {1}".format(os_info.get('NAME', 'Unknown'), os_info.get('VERSION', ''))
        except FileNotFoundError:
            return "Linux"
    elif system == "Windows":
        return "Windows {0}".format(platform.win32_ver()[0])
    elif system == "Darwin":
        return "macOS {0}".format(platform.mac_ver()[0])
    return system.replace('"', '')

def load_history():
    """Load chat history from file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    """Save chat history to file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def clear_history():
    """Clear chat history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    logger.info("Chat history cleared.")

def view_history():
    """View chat history."""
    history = load_history()
    if not history:
        console.print("No chat history.")
        return
    for msg in history:
        role = msg['role'].upper()
        content = msg['content']
        console.print(f"[bold]{role}:[/bold] {content}")

def get_llm_handler(llm):
    """Get LLM-specific request handler."""
    if llm == 'oai':
        return {
            'url': "https://api.openai.com/v1/chat/completions",
            'headers': {"Authorization": f"Bearer {os.getenv('OAI_KEY', config.OAI_KEY)}"},
            'model': config.OAI_LLM
        }
    elif llm == 'ollama':
        return {
            'url': config.OLLAMA_URL + 'api/chat',
            'headers': {},
            'model': config.OLLAMA_LLM
        }
    elif llm == 'grok':
        return {
            'url': "https://api.x.ai/v1/chat/completions",
            'headers': {"Authorization": f"Bearer {os.getenv('GROK_KEY', config.GROK_KEY)}"},
            'model': config.GROK_LLM
        }
    raise ValueError("Invalid LLM")

def stream_api_response(chat_history, args):
    """Stream response from API with improved error handling."""
    llm = args.llm or config.DEFAULT_LLM
    try:
        handler = get_llm_handler(llm)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    payload = {
        "messages": chat_history,
        "stream": True,
        "model": handler['model']
    }

    hist_output = ''
    try:
        response = requests.post(handler['url'], stream=True, json=payload, headers=handler['headers'])
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        return ''

    if not args.x:
        console.log(f'[bold black]{payload["model"]}')

    if args.json_output:
        try:
            full_response = ''.join(line.decode('utf-8') for line in response.iter_lines() if line)
            hist_output = json.loads(full_response)
            console.print(json.dumps(hist_output, indent=4))
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON output.")
        return hist_output

    with (Live(console=console, refresh_per_second=8) if not args.x else nullcontext()) as live:
        for line in response.iter_lines():
            if line:
                try:
                    if llm == 'oai' or llm == 'grok':
                        data = json.loads(line.decode("utf-8")[6:] if line.startswith(b'data: ') else line.decode("utf-8"))
                        char = data['choices'][0]['delta'].get('content', '')
                    else:
                        char = json.loads(line.decode("utf-8"))['message']['content']
                    hist_output += char
                    if not args.x:
                        markdown = Markdown(hist_output)
                        live.update(Panel(markdown))
                    else:
                        print(char, end='', flush=True)
                except (json.JSONDecodeError, KeyError):
                    pass
    if args.x:
        print()
    if not args.x:
        console.log('[bold black]done')
    return hist_output

def extract_command(output):
    """Extract commands from output."""
    command_list = ['bash', 'sh']
    return [snip[len(lang)+1:].strip() for snip in output.split('```') for lang in command_list if snip.strip().startswith(lang)]

def extract_code(output):
    """Extract code from output."""
    code_list = ['python', 'java', 'javascript', 'cpp', 'c', 'ruby', 'html', 'css', 'php', 'sql', 'go', 'rust', 'perl', 'typescript', 'lua']
    return [snip[len(lang)+1:].strip() for snip in output.split('```') for lang in code_list if snip.strip().startswith(lang)]

def build_system_prompt(args, query):
    """Build modular system prompt from config."""
    system_message = SYSTEM_PROMPTS.get('base', []).copy()

    now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
    system_message.append(f"The current date and time is {now}.")

    OS = get_os()
    if OS:
        system_message.append(f"The user operating system is {OS}.")

    if not args.x:
        system_message.append("Your response should be in markdown format.")

    prompt_key = next((k for k in PROMPT_MAP if getattr(args, k, False)), None)
    if prompt_key:
        system_message.extend(SYSTEM_PROMPTS.get(prompt_key, []))

    if args.N:
        system_message.extend(SYSTEM_PROMPTS.get('news', []))

    return ' '.join(system_message)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'history':
        parser = argparse.ArgumentParser(description="AI CLI Assistant - History Management", formatter_class=ColorHelpFormatter)
        parser.add_argument('action', choices=['view', 'clear'], help="Action to perform on history")
        args = parser.parse_args(sys.argv[2:])  # Parse from action onwards

        if args.action == 'view':
            view_history()
        elif args.action == 'clear':
            clear_history()
        sys.exit(0)

    # Main query parser
    parser = argparse.ArgumentParser(description="AI CLI Assistant", formatter_class=ColorHelpFormatter)
    # LLM Provider
    parser.add_argument("--llm", choices=['oai', 'ollama', 'grok'], help="Select LLM provider")

    # Functions
    parser.add_argument("-E", action="store_true", help="Extract command(s) from last output")
    parser.add_argument("-C", action="store_true", help="Extract code from last output")
    parser.add_argument("-N", action="store_true", help="Show the latest news")

    # Formatting
    parser.add_argument("-l", action="store_true", help="Print the last output")
    parser.add_argument("-x", action="store_true", help="Remove formatting from output")
    parser.add_argument("--json-output", action="store_true", help="Output in JSON format")

    # Engineering prompts
    for flag, desc in PROMPT_MAP.items():
        parser.add_argument(f"--{flag}", action="store_true", help=desc)

    parser.add_argument("query", nargs="*", help="Query for the AI")

    args = parser.parse_args()

    # Handle last output actions
    if args.l or args.E or args.C:
        try:
            with open(LAST_OUTPUT_FILE, 'r') as f:
                output = f.read()
            if args.C:
                for c in extract_code(output):
                    print(c)
            elif args.E:
                for c in extract_command(output):
                    print(c)
            elif args.x:
                print(output)
            else:
                console.print(Markdown(output))
        except FileNotFoundError:
            logger.error("No last output found.")
        sys.exit(0)

    # Piped input
    piped_input = sys.stdin.read().strip() if not sys.stdin.isatty() else None

    # Construct query
    if piped_input:
        query = f"<content>{piped_input}</content>\n\n" + " ".join(args.query)
    elif args.N:
        try:
            resp = requests.get(config.NEWS).text
            query = f"<news>{resp}</news>\n\nSummarize the news articles provided."
        except requests.RequestException:
            logger.error("Failed to fetch news feed.")
            sys.exit(1)
    else:
        query = " ".join(args.query)

    if not query and not args.N:
        parser.print_help()
        sys.exit(0)

    system_content = build_system_prompt(args, query)

    history = load_history()
    chat_history = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": query}]

    if DEBUG:
        logger.debug("SYSTEM PROMPT: %s", system_content)

    output = stream_api_response(chat_history, args)

    if output:
        with open(LAST_OUTPUT_FILE, 'w') as f:
            f.write(output)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": output})
        save_history(history)

    # Tool integration example (web search placeholder)
    if 'web search' in query.lower():
        logger.info("Web search integration can be added here.")

if __name__ == "__main__":
    main()
