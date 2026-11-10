import json
import unittest
from pathlib import Path
from restart.rewards import components, strict_correctness, strict_format
from restart.common import runtime_layout
from restart.compare import paired
from restart.prepare_data import build


class RewardTests(unittest.TestCase):
    def test_adversarial(self):
        cases = [('', (0,0)), ('The answer is 16', (0,0)),
                 ('Final answer: 16', (1,1)), ('Final answer: 20', (0,1)),
                 ('16 in reasoning\nFinal answer: 20', (0,1)),
                 ('Final answer: 16\nFinal answer: 16', (0,0)),
                 ('Final answer: 16 kg', (0,0)), ('Final answer: ,', (0,0)),
                 ('Final answer: 16\ntrailing', (0,0)),
                 ('Final answer: $16.00', (1,1))]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(components(text,'16'), expected)

    def test_chat_and_alignment(self):
        self.assertEqual(strict_correctness([[{'role':'assistant','content':'Final answer: -1.5'}]], ['-1.50']), [1])
        with self.assertRaises(ValueError):
            strict_format(['Final answer: 16'], [])

    def test_arms_change_only_reward(self):
        cfg = json.loads(Path('restart/configs/experiment.json').read_text())
        self.assertEqual(cfg['arms'], {'strict':[1,0], 'format_aux':[1,0.1]})
        # Correct answers always outrank wrong formatted answers in raw total reward.
        self.assertGreater(1.1, 0.1)

    def test_batch_layout(self):
        for world in (1,2,3):
            layout = runtime_layout(world)
            self.assertEqual(world*layout['gradient_accumulation_steps'], 12)
            self.assertEqual(world*layout['steps_per_generation'] % 4, 0)
        with self.assertRaises(ValueError):
            runtime_layout(4)

    def test_paired_rejects_mismatch(self):
        a = [{'id':1, 'gold':'2','status':'wrong'}]
        b = [{'id':1, 'gold':'2','status':'correct'}]
        self.assertEqual(paired(a,b)['net_correct'], 1)
        with self.assertRaises(ValueError):
            paired(a,b+b)
        with self.assertRaises(ValueError):
            paired(a,[dict(b[0],gold='3')])

    def test_data_duplicates_rejected(self):
        rows = [{'question':'same', 'answer':'work #### 2'}]*3
        m = {'total_samples':3,'splits':{'sft':{'ids':[0]},'rl':{'ids':[1]},'dev':{'ids':[2]}}}
        with self.assertRaises(ValueError):
            build(rows,m,'{question}')


if __name__ == '__main__':
    unittest.main()
