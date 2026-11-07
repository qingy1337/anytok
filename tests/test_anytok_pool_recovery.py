"""Recover stalled uploads without duplicate GPU work or partial checkpoints."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
controller=importlib.import_module('run_anytok_pool_20k')
transfer=importlib.import_module('modal_anytok_pool_inputs')


class RecoveryTest(unittest.TestCase):
    def test_only_private_quota_errors_switch_to_volume_transfer(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(controller,'STATE',Path(tmp)),patch.object(controller,'emit'):
            path=Path(tmp)/'data-pipelines-upload.log'
            path.write_text('Private storage limit reached for user')
            modes={'transfer':'hf'}
            with patch.object(controller,'call',side_effect=RuntimeError('upload failed')):
                controller.publish('data-pipelines',modes)
            self.assertEqual(modes['transfer'],'modal-volume')
            with patch.object(controller,'call') as call:
                controller.publish('data-pipelines',modes)
                call.assert_not_called()
            path.write_text('Checkpoint hash mismatch')
            modes={'transfer':'hf'}
            with patch.object(controller,'call',side_effect=RuntimeError('invalid checkpoint')):
                with self.assertRaises(RuntimeError):controller.publish('data-pipelines',modes)
            self.assertEqual(modes['transfer'],'hf')

    def test_incomplete_or_weights_only_checkpoints_cannot_be_handed_off(self):
        names=['hypernet/config.json','hypernet/model.safetensors','lm.pt','optimizer.pt','state.pt']
        manifest={'step':8277,'kind':'full_restart_checkpoint',
                  'files':{name:{'bytes':1,'sha256':'a'*64} for name in names}}
        items=transfer.checkpoint_items(manifest,'run/step-008277')
        self.assertEqual(len(items),5)
        manifest['files'].pop('optimizer.pt')
        with self.assertRaises(ValueError):transfer.checkpoint_items(manifest,'run/step-008277')
        manifest['kind']='weights_only_checkpoint'
        with self.assertRaises(ValueError):transfer.checkpoint_items(manifest,'run/step-008277')

    def test_active_gpu_prevents_a_second_controller(self):
        def query(args,**kwargs):
            rows=[{'app_id':'ap-live','description':'anytok-pool-20k-b300',
                   'state':'ephemeral (detached)','tasks':'1'}] if kwargs['env']['MODAL_PROFILE']=='csproject' else []
            return type('Result',(),{'stdout':json.dumps(rows)})()
        with patch.object(controller.subprocess,'run',side_effect=query):
            with self.assertRaises(controller.RecoveryRefused):controller.assert_no_active_training()


if __name__=='__main__':unittest.main()
