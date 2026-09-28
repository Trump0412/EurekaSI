import pytest
from spatial_intelligence.boolean_metrics import summarize
from spatial_intelligence.boolean_metrics import paired_format_analysis


def test_majority_prediction_does_not_improve_balanced_accuracy():
    rows=[{'dataset':'spar','id':str(i),'answer':label} for i,label in enumerate(['No','No','No','Yes'])]
    details=[{'id':'spar::'+str(i),'correct':i<3} for i in range(4)]
    report=summarize(rows,details)
    assert report['accuracy']==.75
    assert report['balanced_accuracy']==.5
    assert report['always_no_accuracy']==.75
    with pytest.raises(ValueError,match='coverage'):summarize(rows,details[:-1])


def test_paired_diagnostic_separates_previously_invalid_outputs():
    before=[{'id':'a','parsed':'no','correct':True},{'id':'b','parsed':'long explanation','correct':False}]
    after=[{'id':'a','parsed':'yes','correct':False},{'id':'b','parsed':'yes','correct':True}]
    result=paired_format_analysis(before,after)
    assert result['baseline_valid_yes_no_subset']=={'n':1,'before_correct':1,'after_correct':0}
    assert result['baseline_invalid_yes_no_subset']=={'n':1,'before_correct':0,'after_correct':1}
    assert result['incorrect_to_correct']==result['correct_to_incorrect']==1
