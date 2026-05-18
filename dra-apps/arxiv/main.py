#!/usr/bin/env python
"""
Deep Orchestrator ArXiv Research Application

This application demonstrates the Deep Orchestrator (AdaptiveOrchestrator) for research of ArXiv papers with:
- Dynamic agent creation and caching
- Knowledge extraction and accumulation
- Budget tracking (tokens, cost, time)
- Task queue management with dependencies
- Policy-driven execution control
- Full state visibility throughout execution
- Cross-session memory via MemMachine (MCP)
"""

import argparse, asyncio, re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from dra.core.common.observer import Observer
from dra.core.common.tasks import BaseTask, GenerateTask, AgentTask
from dra.core.common.utils.io import UserPrompts
from dra.core.common.utils.main import ParserUtil, Runner
from dra.core.common.utils.paths import resolve_path, resolve_and_require_path
from dra.core.common.variables import Variable

def get_server_list() -> list[str]:
    return [
        "fetch",
        "filesystem",
        "arxiv-mcp",
        "docling",
        "memmachine",
    ]

def get_extra_observers() -> dict[str, Observer]:
    return {}

class ArXivParserUtil(ParserUtil):
    def __init__(self, which_app: str, app_name: str, ux_title: str, description: str):
        super().__init__(which_app, app_name, ux_title, description)

    def _do_prompt_for_missing_args(self, up: UserPrompts, args: argparse.Namespace) -> dict[str, Any]:
        query = args.query
        if not query or not query.strip():
            query = up.read_multi_line_input("Input the query for your research")
        
        categories = args.categories
        if not categories or not categories.strip():
            categories = up.read_one_line_input("Input any comma-separated ArXiv categories (https://arxiv.org/category_taxonomy), if any, to focus on",
                empty_allowed=True)
        return {
            'query': query,
            'categories': categories,
        }


def define_cli_arguments() -> ParserUtil:
    def_arxiv_research_agent_prompt_file = "arxiv_research_agent.md"
    
    which_app   = "arxiv"
    app_name    = "arxiv_deep_research"
    ux_title    = "ArXiv Deep Research Agent"
    description = "ArXiv Deep Research using orchestrated AI agents"
    parser_util = ArXivParserUtil(which_app, app_name, ux_title, description)

    parser_util.parser.add_argument(
        "-q", "--query",
        help=f"A quoted string with your research query. If not provided on the command line, you will be prompted for it."
    )
    parser_util.parser.add_argument(
        "--categories",
        help=f"Optional, comma-separated ArXiv categories (see https://arxiv.org/category_taxonomy). Spaces are allowed within the list."
    )
    parser_util.add_arg_markdown_report_path()
    parser_util.add_arg_markdown_research_report_title()
    parser_util.add_arg_output_dir()
    parser_util.add_arg_templates_dir()
    parser_util.parser.add_argument(
        "--arxiv-research-prompt-path",
        default=def_arxiv_research_agent_prompt_file,
        help=f"Path where the main research agent prompt file is located. (Default: {def_arxiv_research_agent_prompt_file}) {parser_util.read_relative_from('templates-dir')}"
    )
    parser_util.add_arg_markdown_yaml_header_template_path()
    parser_util.add_arg_research_model()
    parser_util.add_arg_provider()
    parser_util.add_arg_mcp_agent_config_path()
    parser_util.add_arg_temperature()
    parser_util.add_arg_max_iterations()
    parser_util.add_arg_max_tokens()
    parser_util.add_arg_max_cost_dollars()
    parser_util.add_arg_max_time_minutes()
    parser_util.add_arg_short_run()
    parser_util.add_arg_verbose()
    
    return parser_util

def process_cli_arguments(parser_util: ParserUtil):
    processed_args = parser_util.process_args()

    output_dir_path = processed_args['output_dir_path']    
    templates_dir_path = processed_args['templates_dir_path']
    arxiv_research_prompt_path = resolve_and_require_path(
        processed_args['arxiv_research_prompt_path'], templates_dir_path)

    processed_args['arxiv_research_prompt_path'] = \
        arxiv_research_prompt_path

def create_variables(parser_util: ParserUtil) -> dict[str, Variable]:
    variables_list = [
        Variable("start_time",    parser_util.processed_args['start_time']),
        Variable("query",         parser_util.processed_args['query'], kind='str'),
        Variable("categories",      parser_util.processed_args['categories'], kind='str'),
        Variable("research_report_title",  parser_util.processed_args['research_report_title'], kind='str'),
    ]
    variables_list.extend(parser_util.common_variables())
    
    variables_list.extend([
        Variable("arxiv_research_prompt_path", parser_util.processed_args["arxiv_research_prompt_path"], kind='file'),
    ])
    variables_list.extend(parser_util.only_verbose_common_vars())

    return dict([(v.key, v) for v in variables_list])

def make_tasks(parser_util: ParserUtil, variables: dict[str, Variable]) -> Sequence[BaseTask]:
    tasks = [
        GenerateTask(
            name="arxiv_research",
            title="\U0001F4CA ArXiv Research Result",
            model_name=variables['research_model'].value,
            prompt_template_path=variables['arxiv_research_prompt_path'].value,
            output_dir_path=variables['output_dir_path'].value,
            properties=variables),
    ]
    return tasks

if __name__ == "__main__":
    parser_util = define_cli_arguments()
    process_cli_arguments(parser_util)
    variables = create_variables(parser_util)
    tasks = make_tasks(parser_util, variables)
    runner = Runner(
        tasks, get_server_list(), get_extra_observers(), parser_util, variables)
    asyncio.run(runner.run())
