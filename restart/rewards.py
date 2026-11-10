"""Use the frozen legacy parser; never search reasoning for the gold number."""
from decimal import InvalidOperation
from scripts.evaluator import extract_answer, parse_reference


def as_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return ''.join(message['content'] for message in completion)
    raise TypeError('Expected text or conversational completion')


def components(completion, gold):
    text = as_text(completion)
    try:
        predicted = extract_answer(text)
    except InvalidOperation:
        predicted = None
    reference = parse_reference(str(gold))
    if not reference.is_finite():
        raise ValueError('Nonfinite gold')
    return float(predicted == reference), float(predicted is not None)


def strict_correctness(completions, gold, **kwargs):
    if len(completions) != len(gold):
        raise ValueError('Completion/gold count mismatch')
    return [components(c, g)[0] for c, g in zip(completions, gold)]


def strict_format(completions, gold, **kwargs):
    if len(completions) != len(gold):
        raise ValueError('Completion/gold count mismatch')
    return [components(c, g)[1] for c, g in zip(completions, gold)]
