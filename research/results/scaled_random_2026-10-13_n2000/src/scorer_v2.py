"""Secondary scorer. Never invoked on old saved outputs."""
import re
from decimal import Decimal, InvalidOperation

NUMBER = r'[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?'
LINE = re.compile(r'[ \t]*(?P<currency>[$¥£€₣])?[ \t]*(?P<number>' + NUMBER + r')[ \t]*(?P<unit>%|percent|dollars|pounds)?[ \t]*')

def parse_v2(text):
    if '####' not in text:
        return {'prediction': None, 'continuation': False, 'presentation': False, 'reason': 'missing_marker'}
    suffix = text.split('####', 1)[1]
    first, sep, rest = suffix.partition('\n')
    first = first.removesuffix('\r')
    continuation = bool(sep and rest.strip())
    match = LINE.fullmatch(first)
    if match is None:
        return {'prediction': None, 'continuation': continuation, 'presentation': False, 'reason': 'invalid_answer_line'}
    value = Decimal(match['number'].replace(',', ''))
    value = Decimal(0) if value == 0 else value
    prediction = format(value.normalize(), 'f')
    return {'prediction': prediction, 'continuation': continuation,
            'presentation': bool(match['currency'] or match['unit']), 'reason': None}

def score_v2(text, reference):
    result = parse_v2(text)
    try:
        result['correct'] = result['prediction'] is not None and Decimal(result['prediction']) == Decimal(reference)
    except InvalidOperation:
        result['correct'] = False
    return result
