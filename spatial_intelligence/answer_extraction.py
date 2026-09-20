"""Deterministic, ground-truth-blind final-answer extraction, version 1.

Never search for a gold answer among mentioned candidates. Ambiguity is an abstention.
This is an adapted reporting protocol, not the upstream first-token scorer.
"""
import json
import math
import re

VERSION='final-answer-v1'
NUMBER=r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?'


def extract_answer(response, *, choices=None, truncated=False):
    text=str(response).strip()
    def result(value,status,method):
        return {'answer':value,'status':status,'method':method,'version':VERSION,'truncated':truncated}
    if text.count('<think>')>text.count('</think>'):
        return result(None,'unfinished_thinking','none')
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    # Ignore transport terminators, not arbitrary model prose.
    text=re.sub(r'<\|(?:im_end|endoftext)\|>','',text).strip()
    def scalar(value):
        value=str(value).strip().strip('`*').strip()
        if choices:
            letters=''.join(re.escape(k) for k in choices)
            m=re.fullmatch(r'\(?(['+letters+r'])\)?[.!]?',value,re.I)
            if m:return m.group(1).upper()
            # A labelled answer may include its matching option text, never a contradictory label.
            m=re.fullmatch(r'\(?(['+letters+r'])\)?[.):]\s*(.+)',value,re.I)
            if m and m.group(2).strip().rstrip('.').casefold()==str(choices[m.group(1).upper()]).strip().rstrip('.').casefold():
                return m.group(1).upper()
            matches=[k for k,v in choices.items() if value.rstrip('.').casefold()==str(v).strip().rstrip('.').casefold()]
            return matches[0] if len(matches)==1 else None
        m=re.fullmatch('('+NUMBER+r')(?:\s*(?:cm|m|mm|meters?|metres?|centimeters?|centimetres?|square meters?|m²|m\^2|objects?))?\.?',value,re.I)
        if not m:return None
        value=m.group(1).replace(',','')
        return value if math.isfinite(float(value)) else None
    candidates=[];method='none'
    tagged=re.findall(r'<answer>\s*(.*?)\s*</answer>',text,flags=re.S|re.I)
    boxed=re.findall(r'\\boxed\{([^{}]+)\}',text)
    if tagged:candidates=tagged;method='answer_tag'
    elif boxed:candidates=boxed;method='boxed'
    else:
        try:
            obj=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text))
            if isinstance(obj,dict) and 'answer' in obj:candidates=[obj['answer']];method='json_answer'
        except (ValueError,TypeError):pass
    if not candidates:
        finals=re.findall(r'(?:^|\n|(?<=[.!?])\s+)(?:the\s+)?final answer\s*(?:is\s*|[:=]\s*)([^\n]+)',text,re.I)
        answers=re.findall(r'(?:^|\n|(?<=[.!?])\s+)(?:the\s+)?answer\s*(?:is\s*|[:=]\s*)([^\n]+)',text,re.I)
        if finals:candidates=finals;method='final_answer_statement'
        elif answers:candidates=answers;method='answer_statement'
    if not candidates:
        direct=scalar(text)
        if direct is not None:return result(direct,'ok','direct')
        return result(None,'truncated_no_answer' if truncated else 'no_unambiguous_answer','none')
    values=[scalar(x) for x in candidates]
    if None in values or len(set(values))!=1:return result(None,'ambiguous','explicit')
    return result(values[0],'ok',method)
