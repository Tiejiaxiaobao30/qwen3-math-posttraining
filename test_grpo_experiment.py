"""新增入口的边界测试；微型真实训练验证单独保留运行日志。"""
import json
import pytest
import torch
import train_grpo_experiment as exp
from common import digest, write


def test_ablation_factors_are_explicit():
    base = exp.arm_config('baseline')
    expected = {'no_std':{'scale_rewards'}, 'token_norm':{'loss_type'},
        'token_no_std':{'loss_type','scale_rewards'}, 'dr_grpo':{'loss_type','scale_rewards'},
        'low_lr':{'learning_rate'}}
    for arm,keys in expected.items():
        cfg = exp.arm_config(arm)
        assert {k for k in base if base[k] != cfg[k]} == keys
        assert cfg['num_generations'] == 4 and cfg['beta'] == .04
        assert cfg['max_completion_length'] == 256


def test_single_card_splits_rollouts_without_shrinking_optimizer_batch():
    single = exp.batch_config(1, 'baseline')
    assert single == dict(per_device_train_batch_size=4, gradient_accumulation_steps=3, steps_per_generation=1)
    assert single['gradient_accumulation_steps'] % single['steps_per_generation'] == 0
    assert exp.batch_config(1, 'token_norm')['steps_per_generation'] == 3
    for world in (2,3):
        batch = exp.batch_config(world, 'baseline')
        assert world * batch['steps_per_generation'] == world * batch['gradient_accumulation_steps'] == 12


def test_strict_reward_and_format_diagnostics():
    text = ['Final answer: 7', 'Final answer: 8', '7', 'Final answer: 7\nFinal answer: 7', 'Final answer: ,']
    assert exp.strict_reward(text, ['7']*5) == [1.,0.,0.,0.,0.]
    assert [exp.format_ok(t) for t in text] == [1.,1.,0.,0.,0.]
    result = exp.evaluation_result(dict(id=7,gold='7'),'Final answer: ,',3,False)
    assert result['status']=='format_fail' and result['parsed_answer'] is None
    assert exp.evaluation_result(dict(id=7,gold='7'),'Final answer: 7',3,False)['status']=='correct'


def test_complete_global_groups_and_lengths():
    rewards = torch.tensor([0.,0.,0.,0., 1.,1.,1.,1., 1.,0.,1.,0.])
    ids = torch.tensor([10]*4+[11]*4+[12]*4)
    stats = exp.group_summary(rewards,torch.ones(12),torch.arange(1,13),torch.zeros(12),ids)
    assert stats['mixed_groups'] == pytest.approx(1/3)
    assert stats['all_wrong_groups'] == pytest.approx(1/3)
    assert stats['all_correct_groups'] == pytest.approx(1/3)
    assert stats['correctness'] == .5 and stats['format_fail'] == 0
    exp.require_active_completions(torch.zeros(12))
    with pytest.raises(RuntimeError,match='no active loss tokens'):
        exp.require_active_completions(torch.ones(12))
    ids[-1] = 999
    with pytest.raises(ValueError,match='different question'):
        exp.group_summary(rewards,torch.ones(12),torch.ones(12),torch.zeros(12),ids)


def test_selected_data_cannot_rewrite_or_leak_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(exp,'ROOT',tmp_path)
    source = [dict(id=i,prompt=f'question{i}',gold=str(i)) for i in (1,2,3)]
    write(tmp_path/'data/rl.jsonl',source)
    (tmp_path/'data/manifest.json').write_text(json.dumps(dict(files={'rl':{'sha256':digest(tmp_path/'data/rl.jsonl')}})))
    (tmp_path/'split_manifest.json').write_text(json.dumps(dict(splits={'rl':{'ids':[1,2,3]}})))
    assert exp.load_training_rows(tmp_path/'data/rl.jsonl') == source
    write(tmp_path/'selection/informative.jsonl',source[:1])
    (tmp_path/'selection/run.json').write_text('{}')
    (tmp_path/'selection/rollouts.jsonl').write_text('{}\n')
    selected = dict(complete=True,source_sha256=digest(tmp_path/'data/rl.jsonl'),count=1,
        files={'informative.jsonl':digest(tmp_path/'selection/informative.jsonl')},
        run_sha256=digest(tmp_path/'selection/run.json'),rollouts_sha256=digest(tmp_path/'selection/rollouts.jsonl'))
    (tmp_path/'selection/manifest.json').write_text(json.dumps(selected))
    assert exp.load_training_rows(tmp_path/'selection/informative.jsonl') == source[:1]
    (tmp_path/'selection/informative.jsonl').write_text(json.dumps(dict(id=1,prompt='injected gold',gold='1'))+'\n')
    with pytest.raises(ValueError,match='rewritten'):
        exp.load_training_rows(tmp_path/'selection/informative.jsonl')


def test_no_std_does_not_create_signal_in_constant_groups():
    rewards = torch.tensor([[0.,0.,0.,0.],[1.,1.,1.,1.],[1.,1.,1.,0.]])
    advantage = rewards - rewards.mean(1,keepdim=True)
    assert torch.equal(advantage[:2],torch.zeros(2,4))
    assert torch.equal(advantage[2],torch.tensor([.25,.25,.25,-.75]))


def test_screen_writes_paired_transitions_and_matched_training_data(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = [dict(id=i, prompt=f'q{i}', gold='7') for i in range(6)]
    write(tmp_path/'data/rl.jsonl', source)
    data = tmp_path/'data/rl.jsonl'
    (tmp_path/'data/manifest.json').write_text(json.dumps(dict(files={'rl':{'sha256':digest(data)}})))
    (tmp_path/'split_manifest.json').write_text(json.dumps(dict(splits={'rl':{'ids':list(range(6))}})))
    monkeypatch.setattr(exp, 'ROOT', tmp_path)
    monkeypatch.setattr(exp, 'provenance', lambda: {})
    monkeypatch.setattr(exp, 'assets', lambda model,sft=None: dict(model=model, sft=sft))
    model = SimpleNamespace()
    model.cuda = model.eval = lambda: model
    monkeypatch.setattr(exp, 'load_model', lambda *args: (model,None))
    def rollout(model, tokenizer, rows, seed, stream):
        for i,row in enumerate(rows):
            n = 2 if stream.name.endswith('instruct_probe.jsonl') else [2,2,2,4,0,4][i]
            group = dict(id=row['id'], rewards=[1]*n+[0]*(4-n), format_fail=0, truncated=0)
            stream.write(json.dumps(group)+'\n')
            yield group
    monkeypatch.setattr(exp, 'screening_groups', rollout)
    out = tmp_path/'screen'
    exp.screen(SimpleNamespace(model='base',sft='sft42',instruct='instruct',data=data,seed=42),source,out)
    assert exp.load_training_rows(out/'informative.jsonl') == source[:3]
    assert len(exp.load_training_rows(out/'random.jsonl')) == 3
    report = json.loads((out/'paired_probe.json').read_text())
    assert report['transitions'] == {'mixed->mixed':3, 'mixed->all_correct':2, 'mixed->all_wrong':1}
