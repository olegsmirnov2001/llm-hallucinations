from transformers import PreTrainedTokenizer


def extract_chain_of_thought(
    generated_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool = True,
    *,
    end_think_token: str = '</think>',
) -> tuple[str, str, int]:
    '''
    Returns (chain_of_thought, answer, thinking_duration)
    '''

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True).strip()

    if not enable_thinking:
        answer = decode(generated_ids)
        return '', answer, 0

    end_think_token_id = tokenizer.added_tokens_encoder[end_think_token]
    if end_think_token_id not in generated_ids:
        chain_of_thought = decode(generated_ids)
        return chain_of_thought, '', len(generated_ids)

    thinking_duration = generated_ids.index(end_think_token_id) + 1

    chain_of_thought = tokenizer.decode(generated_ids[:thinking_duration], skip_special_tokens=True).strip()
    answer = tokenizer.decode(generated_ids[thinking_duration:], skip_special_tokens=True).strip()

    return chain_of_thought, answer, thinking_duration
