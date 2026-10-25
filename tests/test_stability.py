"""Small deterministic checks; no corpus training or existing checkpoint writes."""
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import torch
from amc_model.model import AMCTransformer, ModelConfig
from amc_model.run_lock import RunLock
from amc_model.token_data import EpochBatches


class StabilityChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.cfg=ModelConfig(vocab_size=23,context_length=8,n_layers=1,n_heads=2,d_model=16)

    def test_causal_mask_hides_future(self):
        m=AMCTransformer(self.cfg).eval()
        a=torch.tensor([[1,2,3,4,5]])
        b=torch.tensor([[1,2,3,8,9]])
        with torch.inference_mode():
            la,_=m(a);lb,_=m(b)
        torch.testing.assert_close(la[:,:3],lb[:,:3],rtol=0,atol=0)

    def test_learning_rate_schedule(self):
        from amc_model.schedule import learning_rate_at
        # Defaults reproduce the old constant rate.
        self.assertEqual([learning_rate_at(i,100,3e-4) for i in (0,50,99)],[3e-4]*3)
        warm=[learning_rate_at(i,100,1.0,warmup=10,min_ratio=0.1) for i in range(10)]
        self.assertEqual(warm,[(i+1)/10 for i in range(10)])
        self.assertAlmostEqual(learning_rate_at(10,100,1.0,warmup=10,min_ratio=0.1),1.0)
        self.assertAlmostEqual(learning_rate_at(99,100,1.0,warmup=10,min_ratio=0.1),0.1,places=3)
        mid=learning_rate_at(55,100,1.0,warmup=10,min_ratio=0.0)
        self.assertTrue(0.4<mid<0.6, mid)
        with self.assertRaises(ValueError):
            learning_rate_at(0,100,1.0,min_ratio=2.0)

    def test_padding_targets_are_ignored(self):
        m=AMCTransformer(self.cfg).eval()
        x=torch.tensor([[1,2,3,4]])
        y=torch.tensor([[2,3,-100,-100]])
        logits,loss=m(x,y)
        expected=torch.nn.functional.cross_entropy(logits[:,:2].reshape(-1,23),y[:,:2].reshape(-1))
        torch.testing.assert_close(loss,expected)

    def test_epoch_coverage_and_restart_order(self):
        for length in (2,8,9,34):
            ids=np.arange(length,dtype='<u2')
            a=EpochBatches(ids,8,3,42);b=EpochBatches(ids,8,3,42)
            found=[]
            for i in range(len(a)):
                x,y=a.batch(i);_,other=b.batch(i)
                np.testing.assert_array_equal(y,other)
                np.testing.assert_array_equal((x+1)[y!=-100],y[y!=-100])
                found.extend(y[y!=-100].tolist())
            self.assertEqual(sorted(found),list(range(1,length)))

    def test_optimizer_resume_matches_uninterrupted_updates(self):
        m=AMCTransformer(self.cfg)
        optimizer=torch.optim.AdamW(m.parameters(),lr=3e-4)
        x=torch.tensor([[1,2,3,4]]);y=torch.tensor([[2,3,4,5]])
        def update(model,opt):
            opt.zero_grad(set_to_none=True)
            _,loss=model(x,y);loss.backward();opt.step()
        update(m,optimizer)
        buffer=io.BytesIO()
        torch.save({'model':m.state_dict(),'optimizer':optimizer.state_dict()},buffer)
        update(m,optimizer)
        buffer.seek(0)
        state=torch.load(buffer,weights_only=True)
        restored=AMCTransformer(self.cfg)
        restored.load_state_dict(state['model'])
        opt=torch.optim.AdamW(restored.parameters(),lr=3e-4)
        opt.load_state_dict(state['optimizer'])
        update(restored,opt)
        for key,value in m.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
        self.assertIs(restored.token_embeddings.weight,restored.lm_head.weight)

    def test_run_directory_allows_only_one_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            first=RunLock(folder)
            try:
                with self.assertRaisesRegex(RuntimeError,'Another trainer'):
                    RunLock(folder)
            finally:first.close()
            second=RunLock(folder);second.close()


if __name__=='__main__':unittest.main()
