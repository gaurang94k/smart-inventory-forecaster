from __future__ import annotations

import json
import os
import time

import anthropic

from src.agent.tools import (
    get_forecast_summary,
    get_inventory_recommendation,
    list_skus,
    run_service_level_scenario,
)

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are an inventory planning assistant for a retail forecasting system. "
    "You have access to tools that can retrieve forecast performance metrics, inventory "
    "recommendations, and run what-if scenarios. Always use tools to get accurate data "
    "before answering. Be concise and business-focused in your responses."
)

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "list_skus",
        "description": "Return the list of 5 SKU-store IDs available in the forecasting system.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_forecast_summary",
        "description": (
            "Return mean MAPE, RMSE, and MASE at a 28-day horizon for the best forecasting model "
            "for a given SKU. Use this to understand forecast accuracy before making inventory decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_store": {
                    "type": "string",
                    "description": "The SKU-store identifier, e.g. 'FOODS_3_090_CA_1'.",
                },
            },
            "required": ["sku_store"],
        },
    },
    {
        "name": "get_inventory_recommendation",
        "description": (
            "Return the current inventory recommendation for a SKU, including safety stock, "
            "reorder point, EOQ, and annual cost. Based on a 95% service level and 7-day lead time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_store": {
                    "type": "string",
                    "description": "The SKU-store identifier, e.g. 'FOODS_3_090_CA_1'.",
                },
            },
            "required": ["sku_store"],
        },
    },
    {
        "name": "run_service_level_scenario",
        "description": (
            "Re-run the inventory optimizer for a SKU at a different service level and return "
            "the new recommendation plus the delta vs the current 95% baseline. Use this for "
            "what-if analysis (e.g. 'what happens if we raise service level to 99%?')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_store": {
                    "type": "string",
                    "description": "The SKU-store identifier, e.g. 'FOODS_3_090_CA_1'.",
                },
                "service_level": {
                    "type": "number",
                    "description": "Target service level as a decimal between 0 and 1, e.g. 0.99.",
                },
            },
            "required": ["sku_store", "service_level"],
        },
    },
]

TOOL_FUNCTIONS = {
    "list_skus": list_skus,
    "get_forecast_summary": get_forecast_summary,
    "get_inventory_recommendation": get_inventory_recommendation,
    "run_service_level_scenario": run_service_level_scenario,
}


def _dispatch_tool(name: str, inputs: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**inputs)
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _create_message(client: anthropic.Anthropic, messages: list[dict]) -> anthropic.types.Message:
    """Call the API with exponential-backoff retry on 529 overloaded errors.

    persistent 529 errors are retried up to 3 times with backoff
    (1 s, 2 s, 4 s) before the exception is re-raised to the caller.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        except anthropic.APIStatusError as exc:
            if exc.status_code == 529 and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1 s, 2 s, 4 s
                time.sleep(wait)
                continue
            raise  # re-raise on final attempt or non-529 errors


def run_agent(user_message: str, max_turns: int = 5) -> str:
    """Run the inventory planning agent on a user message.

    Loops until the model produces a text-only response or max_turns is reached.
    Returns the final assistant text.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    messages: list[dict] = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        response = _create_message(client, messages)

        # Collect any text from this turn
        text_blocks = [b.text for b in response.content if b.type == "text"]

        # If no tool calls, we're done
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            return "\n".join(text_blocks)

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool call and build the tool_result turn
        tool_results = []
        for block in tool_use_blocks:
            result_content = _dispatch_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_content,
            })

        messages.append({"role": "user", "content": tool_results})

    return "Agent reached max_turns without a final text response."
