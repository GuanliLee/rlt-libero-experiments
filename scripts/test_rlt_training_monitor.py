import copy
import unittest

from rlt_training_monitor import TrainingMonitor


def results(candidate=(True,False), steps=100):
    out=[]
    for policy, mask in [('baseline',(True,False)),('bc_only',(True,False)),('candidate',candidate)]:
        for t,success in enumerate(mask):
            out.append(dict(policy=policy,task_id=t,initial_state=31,success=success,action_steps=steps if success else 300))
    return out


class MonitorTests(unittest.TestCase):
    def make(self,**kw):
        return TrainingMonitor(run_id='fixture',tasks=(0,1),initials=(31,),**kw)

    def observe(self,m,n,r=None):
        return m.observe(run_id='fixture',episode=n,episodes=results() if r is None else r)

    def test_patience_and_initial_checkpoint(self):
        m=self.make()
        with self.assertRaises(ValueError):self.observe(m,30)
        self.assertTrue(self.observe(m,0)['save_best'])
        for n in [30,60,90,120]:self.assertFalse(self.observe(m,n)['stop_requested'])
        self.assertEqual(self.observe(m,150)['stop_reason'],'validation_plateau')
        self.assertEqual(m.best_episode,0)

    def test_success_improvement_resets_patience_and_restore(self):
        m=self.make();self.observe(m,0);self.observe(m,90)
        self.assertTrue(self.observe(m,120,results((True,True)))['save_best'])
        self.assertEqual(m.stalled_evaluations,0)
        restored=TrainingMonitor(**copy.deepcopy(m.state_dict()))
        self.assertEqual(self.observe(m,150,results((True,True))),self.observe(restored,150,results((True,True))))

    def test_speed_cannot_trade_away_success_or_change_survivors(self):
        m=self.make();self.observe(m,0)
        self.assertFalse(self.observe(m,30,results((False,True),50))['save_best'])
        self.assertFalse(self.observe(m,60,results((False,False),1))['save_best'])
        self.assertFalse(self.observe(m,90,results(steps=98))['save_best'])
        self.assertTrue(self.observe(m,120,results(steps=90))['save_best'])

    def test_budget_stops_even_on_improvement(self):
        m=self.make(max_episodes=90);self.observe(m,0)
        r=self.observe(m,90,results((True,True)))
        self.assertTrue(r['save_best']);self.assertEqual(r['stop_reason'],'budget')

    def test_reject_test_missing_duplicate_wrongrun_changedbaseline(self):
        m=self.make();self.observe(m,0)
        for kwargs in [dict(split='test'),dict(run_id='another')]:
            args=dict(run_id='fixture',episode=30,episodes=results());args.update(kwargs)
            with self.assertRaises(ValueError):m.observe(**args)
        for rows in [results()[:-1],results()+[results()[0]]]:
            with self.assertRaises(ValueError):self.observe(m,30,rows)
        rows=results();rows[0]['success']=False
        with self.assertRaises(ValueError):self.observe(m,30,rows)
        self.assertEqual(m.last_episode,0)


if __name__=='__main__':unittest.main()
