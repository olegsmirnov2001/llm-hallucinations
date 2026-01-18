import torch
from transformers import PreTrainedTokenizerBase


def get_soft_stop_thinking_fn(
    tokenizer: PreTrainedTokenizerBase,
    thinking_budget: int,
    input_length: int,
    model_vocab_size: int,
    soft_stop_sequence: str = '... Okay, now I have to answer.\n</think>',
):
    soft_stop_tokens = tokenizer(soft_stop_sequence, return_tensors="pt").input_ids[0]
    end_thinking_token = soft_stop_tokens[-1]

    all_tokens = list(range(model_vocab_size))

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        potential_soft_stop_sequence_position = len(input_ids) - input_length - thinking_budget
        if -len(soft_stop_tokens) <= potential_soft_stop_sequence_position < 0:
            if not (input_ids == end_thinking_token).any():
                only_possible_token = soft_stop_tokens[potential_soft_stop_sequence_position].item()
                return [only_possible_token]
        return all_tokens

    return prefix_allowed_tokens_fn
