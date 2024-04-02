import math
from typing import NamedTuple, final

from langchain.schema.runnable import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input


@final
class ChainOutputWithLogprobs(NamedTuple):
    """
    NamedTuple to store chain output and logprobs.
    """

    output: dict
    logprobs: dict


async def ainvoke_logprobs(
    chain: Runnable, input: Input, config: RunnableConfig | None = None
) -> ChainOutputWithLogprobs:
    """
    Invoke a chain to generate final output plus logprobs data.
    """
    final_output, logprobs = {}, {}
    async for log in chain.astream_log(input, config=config):
        for op in log.ops:
            if op["path"] == "/final_output":
                final_output = op["value"]
            if "/final_output" in op["path"]:
                op_value = op["value"]
                if "generations" in op_value:
                    try:
                        logprobs = op_value["generations"][0][0]["generation_info"]["logprobs"]
                    except (KeyError, IndexError):
                        pass

    return ChainOutputWithLogprobs(output=final_output, logprobs=logprobs)


def compute_total_probability(log_probs: list) -> float:
    # Convert each log probability to a probability and multiply them
    total_prob = math.exp(sum(log_probs))
    return total_prob
