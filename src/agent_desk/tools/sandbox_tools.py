"""Agent-facing tool for exact arithmetic in a restricted Python sandbox."""

import json

from langchain_core.tools import tool

from ..sandbox.python_sandbox import run_code


@tool
def run_python(code: str) -> str:
    """Run a short Python calculation and return what it prints, as JSON.

    Use this for every calculation (ratios, percentages, differences, growth rates, sums,
    averages) instead of calculating yourself. Copy the numbers exactly as the data tools
    returned them, print each result, and use the printed value in your answer. A final bare
    expression is printed automatically.

    The result has ok, output and error. Limits: math, statistics, fractions and decimal only
    (math, statistics, Fraction and Decimal are already available); no files, no network, no
    classes; 5 second time limit; 4000 characters of output. Use f-strings for formatting.
    Example: print(round(88523 / 211915 * 100, 1))
    """
    return json.dumps(run_code(code), ensure_ascii=False, indent=2)
