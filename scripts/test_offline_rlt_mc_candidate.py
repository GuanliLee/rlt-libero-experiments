#!/usr/bin/env python3
"""CPU checks for return reconstruction, mapping rejection, and run guards."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

import offline_rlt_mc_candidate as mc


def fixture():
    config=dict(tasks=[0,1],train_initials=list(range(30)),warmup_episodes=1,train_episodes=2,
                gamma=.99,action_horizon=320,chunk_length=8,reward_scale=5)
    definitions=[(0,'baseline',0,10,True),(1,'baseline',0,8,False),
                 (0,'candidate',1,1,True),(1,'candidate',1,16,True)]
    events,buffer,index=[],[],{0:[],1:[]}
    for task,policy,initial,steps,success in definitions:
        chunks=(steps+7)//8
        start=len(buffer)
        observations=[(torch.full((1,3),float(start+j)),torch.full((8,7),.2),torch.full((8,),float(start+j)/100))for j in range(chunks)]
        for j,(z,ref,prop) in enumerate(observations):
            done=j==chunks-1
            if done:nz,nref,nprop=torch.zeros_like(z),torch.zeros_like(ref),torch.zeros_like(prop)
            else:nz,nref,nprop=observations[j+1]
            reward=5*.99**((steps-1)%8) if done and success else 0.
            buffer.append((z,ref,torch.zeros(8,7),torch.tensor(reward),nz,nref,torch.tensor(float(done)),prop,nprop,task))
            index[task].append(len(buffer)-1)
        events.append(dict(event='episode',collect=True,task_id=task,policy=policy,initial_state=initial,
                           action_steps=steps,chunks=chunks,success=success,reward=5. if success else 0.))
    replay=dict(capacity=100,buffer=buffer,pos=len(buffer),task_index=index)
    progress=dict(warmup_done=2,episodes_done=2,env_steps=sum(e['action_steps']for e in events))
    return replay,events,config,progress


class ReturnTests(unittest.TestCase):
    def test_chunk_discount_and_partial_terminal_are_exact(self):
        data=fixture()
        returns,audit=mc.reconstruct_behavior_returns(*data)
        expected=torch.tensor([5*.99**9,5*.99,0.,5.,5*.99**15,5*.99**7])
        torch.testing.assert_close(returns,expected,atol=1e-6,rtol=1e-6)
        self.assertEqual(audit['episodes'],4)
        self.assertEqual(audit['chunks'],6)
        self.assertTrue(audit['all_terminal_reward_task_mappings_verified'])

    def test_validation_does_not_enter_training_mapping(self):
        replay,events,config,progress=fixture()
        validation=copy.deepcopy(events[0]);validation.update(collect=False,initial_state=40)
        expected,_=mc.reconstruct_behavior_returns(replay,events,config,progress)
        actual,_=mc.reconstruct_behavior_returns(replay,events+[validation],config,progress)
        torch.testing.assert_close(actual,expected)
        events[0]['initial_state']=40
        with self.assertRaisesRegex(ValueError,'schedule'):
            mc.reconstruct_behavior_returns(replay,events,config,progress)

    def test_reject_capacity_overflow_and_missing_chunks(self):
        replay,events,config,progress=fixture()
        replay['capacity']=5
        with self.assertRaisesRegex(ValueError,'capacity overflow'):
            mc.reconstruct_behavior_returns(replay,events,config,progress)
        replay,events,config,progress=fixture();replay['buffer'].pop()
        with self.assertRaisesRegex(ValueError,'length mismatch'):
            mc.reconstruct_behavior_returns(replay,events,config,progress)

    def test_reject_wrong_terminal_reward_task_and_next_observation(self):
        for field,replacement,message in [(6,torch.tensor(1.),'terminal'),(3,torch.tensor(5.),'reward'),
                                           (9,1,'task'),(4,torch.ones(1,3)*99,'next observation')]:
            with self.subTest(field=field):
                replay,events,config,progress=fixture()
                transition=list(replay['buffer'][0]);transition[field]=replacement
                replay['buffer'][0]=tuple(transition)
                with self.assertRaisesRegex(ValueError,message):
                    mc.reconstruct_behavior_returns(replay,events,config,progress)

    def test_reject_corrupt_task_index_and_incomplete_schedule(self):
        replay,events,config,progress=fixture();replay['task_index'][0]=[0]
        with self.assertRaisesRegex(ValueError,'task index'):
            mc.reconstruct_behavior_returns(replay,events,config,progress)
        replay,events,config,progress=fixture();events.pop()
        with self.assertRaisesRegex(ValueError,'episode count'):
            mc.reconstruct_behavior_returns(replay,events,config,progress)

    def test_active_parent_rejected_before_checkpoint_open(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder)/'parent';parent.mkdir()
            (parent/'run_summary.json').write_text(json.dumps(dict(status='running',runner_version=2)))
            # No checkpoint exists; a correct guard fails before trying to read it.
            args=argparse.Namespace(run=parent,output=Path(folder)/'output')
            with self.assertRaisesRegex(ValueError,'fully completed'):
                mc.run(args)
            self.assertFalse(args.output.exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
