"""Class-aware reporting for the spatial yes/no pilot, not official benchmark scores."""


def summarize(rows,details):
    gold={row['dataset']+'::'+row['id']:row['answer'].strip().lower() for row in rows}
    if len(gold)!=len(rows) or set(gold.values())!={'yes','no'}:
        raise ValueError('Unique rows and both Yes/No classes are required')
    by_id={row['id']:row for row in details}
    if len(by_id)!=len(details) or set(gold)!=set(by_id):raise ValueError('Prediction coverage mismatch')
    report={}
    for label in ('yes','no'):
        ids=[sid for sid,value in gold.items() if value==label]
        report[label]={'n':len(ids),'accuracy':sum(by_id[sid]['correct'] for sid in ids)/len(ids)}
    return {'n':len(rows),'accuracy':sum(row['correct'] for row in details)/len(rows),
            'valid_yes_no_rate':sum(row.get('parsed') in {'yes','no'} for row in details)/len(rows),
            'balanced_accuracy':sum(value['accuracy'] for value in report.values())/2,
            'always_no_accuracy':report['no']['n']/len(rows),'always_no_balanced_accuracy':.5,
            'by_answer':report}


def paired_format_analysis(before,after):
    """Post-hoc attribution, not a replacement primary metric or causal proof."""
    first={row['id']:row for row in before};second={row['id']:row for row in after}
    if len(first)!=len(before) or len(second)!=len(after) or first.keys()!=second.keys():
        raise ValueError('Paired coverage mismatch')
    valid=[sid for sid,row in first.items() if row.get('parsed') in {'yes','no'}]
    invalid=[sid for sid in first if sid not in set(valid)]
    def subset(ids):
        return {'n':len(ids),
                'before_correct':sum(bool(first[sid]['correct']) for sid in ids),
                'after_correct':sum(bool(second[sid]['correct']) for sid in ids)}
    return {'baseline_valid_yes_no_subset':subset(valid),
            'baseline_invalid_yes_no_subset':subset(invalid),
            'incorrect_to_correct':sum(not first[sid]['correct'] and second[sid]['correct'] for sid in first),
            'correct_to_incorrect':sum(first[sid]['correct'] and not second[sid]['correct'] for sid in first),
            'scope':'post-hoc diagnostic; formatting and semantic changes are not causally separated'}
