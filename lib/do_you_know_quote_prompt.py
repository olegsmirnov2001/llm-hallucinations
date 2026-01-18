def build_prompt(quote: str) -> str:
    return (
        f'"{quote}" Do you know where is the phrase from? '
        'If you don\'t know, say "No, origin is unclear.". '
        'If you know, say "Yes, the phrase is from <source>.".'
    )
