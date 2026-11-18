from decimal import Decimal

from common import extract_answer, score, build_result


def test_extract_integer():
    assert extract_answer(
        "Reasoning...\nFinal answer: 72"
    ) == Decimal("72")


def test_extract_negative_decimal():
    assert extract_answer(
        "Reasoning...\nFinal answer: -3.5"
    ) == Decimal("-3.5")


def test_extract_comma_number():
    assert extract_answer(
        "Reasoning...\nFinal answer: 1,234"
    ) == Decimal("1234")


def test_extract_dollar_number():
    assert extract_answer(
        "Reasoning...\nFinal answer: $72"
    ) == Decimal("72")


def test_missing_protocol():
    assert extract_answer(
        "I think the answer is 72"
    ) is None


def test_empty_final_answer():
    assert extract_answer(
        "Reasoning...\nFinal answer:"
    ) is None


def test_text_after_number():
    assert extract_answer(
        "Reasoning...\nFinal answer: 72 apples"
    ) is None


def test_multiple_final_answers():
    text = (
        "Final answer: 10\n"
        "I changed my mind.\n"
        "Final answer: 20"
    )

    assert extract_answer(text) is None


def test_score_correct():
    assert score(
        "Reasoning...\nFinal answer: 72",
        "72"
    ) == "correct"


def test_score_wrong():
    assert score(
        "Reasoning...\nFinal answer: 73",
        "72"
    ) == "wrong"


def test_score_format_fail():
    assert score(
        "I think the answer is 72",
        "72"
    ) == "format_fail"


def test_empty_text():
    assert extract_answer("") is None


def test_trailing_spaces():
    assert extract_answer(
        "Reasoning...\nFinal answer: 72   "
    ) == Decimal("72")


def test_reference_with_comma():
    assert score(
        "Reasoning...\nFinal answer: 1234",
        "1,234"
    ) == "correct"


def test_build_result_correct():
    result = build_result(
        sample_id=100,
        prediction="Reasoning...\nFinal answer: 72",
        reference="72",
        generated_tokens=120,
        truncated=False,
    )

    assert result["id"] == 100
    assert result["parsed_answer"] == "72"
    assert result["status"] == "correct"
    assert result["generated_tokens"] == 120
    assert result["truncated"] is False


def test_build_result_wrong():
    result = build_result(
        sample_id=101,
        prediction="Reasoning...\nFinal answer: 73",
        reference="72",
        generated_tokens=80,
        truncated=False,
    )

    assert result["parsed_answer"] == "73"
    assert result["status"] == "wrong"


def test_build_result_format_fail():
    result = build_result(
        sample_id=102,
        prediction="I think the answer is 72",
        reference="72",
        generated_tokens=384,
        truncated=True,
    )

    assert result["parsed_answer"] is None
    assert result["status"] == "format_fail"
    assert result["truncated"] is True

"""CPU checks for split isolation, selection and preservation of raw examples."""
import pytest
from data import build
from sample import select
from train import reward, layout

def test_data_keeps_gold_out_of_prompt_and_isolates_splits():
    raw = [dict(question=f'Question {i}', answer=f'Reasoning <<1+1=2>>\n#### {i+700}') for i in range(3)]
    manifest = dict(total_samples=3, splits={s:dict(ids=[i]) for i,s in enumerate(['sft','rl','dev'])})
    data = build(raw, manifest)
    assert '700' not in data['sft'][0]['prompt']
    assert data['sft'][0]['completion'] == 'Reasoning \nFinal answer: 700'
    assert 'completion' not in data['rl'][0]
    raw[2]['question'] = '  QUESTION 0 '
    with pytest.raises(ValueError, match='Duplicate'):
        build(raw, manifest)

def test_selection_excludes_zero_and_full_correct_and_matches_control_count():
    source = [dict(id=i, prompt=f'q{i}', gold=str(i)) for i in range(5)]
    results = [dict(id=i,rewards=[1]*i+[0]*(4-i)) for i in range(5)]
    chosen, control = select(source, results, 42)
    assert chosen == source[1:4]
    assert len(control) == 3 and len({r['id'] for r in control}) == 3
    assert all(r in source for r in control)
    assert select(source,results,42) == (chosen,control)
    with pytest.raises(ValueError):
        select(source,results[:-1],42)

def test_strict_reward_and_multi_gpu_completion_layout():
    assert reward(['Final answer: 7','7','Final answer: 8'], ['7']*3) == [1.,0.,0.]
    for world in [1,2,3]:
        assert world * layout(world)['gradient_accumulation_steps'] == 12
    with pytest.raises(ValueError):
        layout(4)

def test_paired_analysis_and_mismatch_rejection():
    from evaluate import paired
    before = [dict(id=1, reference='7', status='wrong'), dict(id=2, reference='8', status='correct')]
    after = [dict(id=1, reference='7', status='correct'), dict(id=2, reference='8', status='wrong')]
    result = paired(before, after)
    assert result['improved'] == [1] and result['regressed'] == [2]
    assert result['net_correct'] == 0
    with pytest.raises(ValueError):
        paired(before, after[:1])
    with pytest.raises(ValueError):
        paired(before, [dict(id=1, reference='999', status='correct'), after[1]])
