import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pilot_history_receiver.training_options import requested_updates
from run_memory_training import continued_registration, evaluation_updates
from train_memory_receiver import main as train_main
from train_neural_rx import training_paths
from pilot_history_receiver.checkpoints import recover_history


ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def publish_checkpoint(directory, step):
    checkpoint = Path(directory)/f"checkpoint_{step:06d}"
    checkpoint.mkdir(parents=True)
    for name in ("receiver.weights.h5", "cnn_autoencoder_finalist.json", "state.index", "state.data-00000-of-00001"):
        (checkpoint/name).write_bytes(str(step).encode())
    write_json(checkpoint/"state.json", dict(update=step))
    files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}
    write_json(checkpoint/"manifest.json", dict(update=step,files=files))
    write_json(Path(directory)/"latest.json", dict(update=step,checkpoint=checkpoint.name))
    return checkpoint


class TrainingControlsTest(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads((ROOT/"config/memory_training.json").read_text())

    def test_launcher_registers_retained_anchors_and_development_bank(self):
        import run_memory_training as launcher
        from temporal_generalization_protocol import seal_payload
        from temporal_benchmark_candidate import pretrained_checkpoint
        benchmark = json.loads((ROOT/"config/temporal_benchmark_v1.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            anchors = directory/"anchors.json"
            write_json(anchors, seal_payload(dict(teacher_sha256=launcher.file_sha256(pretrained_checkpoint()),
                prbs=benchmark["radio"]["n_size_bwp"], conditions={}), field="manifest_sha256"))
            dataset = seal_payload(dict(identity=dict(radio=benchmark["radio"])), field="manifest_sha256")
            write_json(directory/"dataset.json", dataset)
            arguments = ["pilot", "--steps", "1000", "--output-dir", str(directory/"results"),
                "--data-gpu", "none", "--replica-gpu", "none", "--anchors", str(anchors),
                "--development-cache", str(directory), "--dry-run"]
            with mock.patch.object(sys, "argv", arguments), mock.patch("builtins.print") as printed, \
                    mock.patch.object(launcher, "recover_latest", side_effect=AssertionError("Dry run must not mutate checkpoints")):
                launcher.main()
            record = json.loads(printed.call_args.args[0])
            self.assertEqual(record["checkpoints"], [1000])
            self.assertEqual(record["calibration_anchors"]["sha256"], launcher.file_sha256(anchors))
            self.assertEqual(record["development_cache"]["manifest_sha256"], dataset["manifest_sha256"])
            self.assertFalse((directory/"results").exists())
            dataset["identity"]["radio"]["n_size_bwp"] += 1
            write_json(directory/"dataset.json", seal_payload(dataset, field="manifest_sha256"))
            with mock.patch.object(sys, "argv", arguments), self.assertRaisesRegex(ValueError, "radio configuration"):
                launcher.main()

    def test_base_training_preserves_pretrained_files_and_existing_outputs(self):
        source = ROOT/"models/pretrained/nrx_large_weights"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initial, destination = training_paths("nrx_large", directory)
            self.assertEqual(initial, source)
            self.assertEqual(destination, directory/"nrx_large_weights")
            for output in (ROOT/"models/pretrained", ROOT/"models/pretrained/subdirectory"):
                with self.assertRaisesRegex(ValueError, "pretrained weights"):
                    training_paths("nrx_large", output)
            alias = directory/"alias"
            alias.symlink_to(ROOT/"models/pretrained", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "pretrained weights"):
                training_paths("nrx_large", alias)
            destination.hardlink_to(source)
            with self.assertRaises(FileExistsError):
                training_paths("nrx_large", directory)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_training_log_recovers_only_updates_in_the_checkpoint(self):
        row = lambda step: json.dumps(dict(update=step,width=12,episode_fingerprints=[str(step)])).encode()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"history.jsonl"
            first = row(1)+b"\n"
            for suffix in (b'{"update": 2, "width":', b'{"update": 2, "note": "\xe2'):
                path.write_bytes(first+suffix)
                recover_history(path, 1)
                self.assertEqual(path.read_bytes(), first)
                with path.open("ab") as stream:
                    stream.write(row(2)+b"\n")
            path.write_bytes(first+row(2))
            recover_history(path,2)
            self.assertEqual(path.read_bytes(),first+row(2)+b"\n")
            path.write_bytes(first+row(2)+b"\n"+row(3)+b"\n")
            recover_history(path,1)
            self.assertEqual(path.read_bytes(),first)
            for contents, step in ((first+b'broken\n',1),(first+b'{"update":',2),
                                   (first+row(1)+b"\n",1)):
                path.write_bytes(contents)
                with self.assertRaises(ValueError):
                    recover_history(path,step)
                self.assertEqual(path.read_bytes(),contents)

    def test_external_resume_at_stop_publishes_a_local_checkpoint(self):
        class Trainer:
            def __init__(self,*args):
                self.seed = 1
                self.value = 0
                self.episodes = SimpleNamespace()
                self.step = SimpleNamespace(numpy=lambda:self.value)
            def binding(self): return {}
            def schedule_state(self): return {}
            def restore(self,path): self.value = 6000
            def save(self,path): return publish_checkpoint(path,self.value)
            def update(self,*args): raise AssertionError("No update is required")
        tensorflow = SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda _:[],
            experimental=SimpleNamespace(enable_op_determinism=lambda:None)))
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ), \
                mock.patch.dict(sys.modules,{"tensorflow":tensorflow,
                    "pilot_history_receiver.training":SimpleNamespace(MemoryTrainer=Trainer,atomic_json=write_json)}):
            source = publish_checkpoint(Path(temporary)/"source",6000)
            output = Path(temporary)/"continued"
            for resume in (source,output/"checkpoint_006000"):
                with mock.patch.object(sys,"argv",["train","--output-dir",str(output),"--data-gpu","none","--replica-gpu","none",
                    "--steps","6000","--resume",str(resume)]):
                    train_main()
                completion = json.loads((output/"completion.json").read_text())
                self.assertEqual(completion["status"],"complete")
                self.assertTrue(Path(completion["checkpoint"]).is_dir())
                self.assertEqual(json.loads((output/"latest.json").read_text())["update"],6000)

    def test_total_updates_are_unrestricted_and_nonnegative(self):
        for steps in (0, 1, 1537, 6001, 1000000):
            self.assertEqual(requested_updates(steps), steps)
        for steps in (None, -1, 1.5, "6000", True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                requested_updates(steps)


    def test_increasing_duration_preserves_the_run_identity(self):
        previous = dict(recipe=self.recipe, requested_updates=1701,
                        checkpoints=evaluation_updates(self.recipe, 1701), architecture={"memory":"pilot_history_attention"})
        current = dict(previous, requested_updates=8001,
                       checkpoints=evaluation_updates(self.recipe, 8001))
        merged = continued_registration(previous, current)
        self.assertEqual(merged["requested_updates"], 8001)
        self.assertIn(1701, merged["checkpoints"])
        changed = copy.deepcopy(current)
        changed["recipe"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "different registration"):
            continued_registration(previous, changed)



    def test_direct_cli_continues_and_recovers_after_update_failure(self):
        restores, updates = [], []
        fail_update = []

        class Trainer:
            def __init__(self, *args):
                self.value = 0
                self.seed = 1
                self.episodes = SimpleNamespace()
                self.step = SimpleNamespace(numpy=lambda:self.value)

            def binding(self):
                return {}

            def schedule_state(self):
                return {}

            def restore(self, path):
                self.value = json.loads((Path(path)/"state.json").read_text())["update"]
                restores.append(self.value)

            def save(self, path):
                return publish_checkpoint(path,self.value)

            def update(self, packets):
                run = json.loads((output/"run.json").read_text())
                self_status = run["status"]
                if self_status != "running" or (output/"completion.json").exists() or (output/"failure.json").exists():
                    raise AssertionError("stale training status while running")
                if fail_update:
                    raise RuntimeError("test update failure")
                self.value += 1
                updates.append(self.value)
                return {"update":self.value}, []

        tensorflow = SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda _:[],
                                    experimental=SimpleNamespace(enable_op_determinism=lambda:None)))
        training_module = SimpleNamespace(MemoryTrainer=Trainer,atomic_json=write_json)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ), \
                mock.patch.dict(sys.modules,{"tensorflow":tensorflow,"pilot_history_receiver.training":training_module}), \
                mock.patch("builtins.print"):
            output = Path(temporary)
            checkpoint = publish_checkpoint(output,5999)
            for stop in (6001, 6003):
                with mock.patch.object(sys,"argv",["train","--output-dir",str(output),"--data-gpu","none","--replica-gpu","none",
                                                   "--steps",str(stop),"--resume",str(checkpoint)]):
                    train_main()
                completion = json.loads((output/"completion.json").read_text())
                self.assertEqual(completion["updates"],stop)
                self.assertEqual(json.loads((output/"run.json").read_text())["status"], "complete")
                checkpoint = Path(completion["checkpoint"])
            self.assertEqual(restores,[5999,6001])
            self.assertEqual(updates,[6000,6001,6002,6003])
            argv = ["train", "--output-dir", str(output), "--data-gpu", "none",
                    "--steps", "6004", "--resume", str(checkpoint)]
            fail_update.append(True)
            with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(RuntimeError, "test update failure"):
                train_main()
            status = json.loads((output/"run.json").read_text())
            self.assertEqual((status["status"], status["last_completed_update"]), ("failed", 6003))
            self.assertTrue((output/"failure.json").is_file())
            self.assertFalse((output/"completion.json").exists())
            fail_update.clear()
            with mock.patch.object(sys, "argv", argv):
                train_main()
            self.assertFalse((output/"failure.json").exists())
            self.assertEqual(json.loads((output/"run.json").read_text())["status"], "complete")

            with mock.patch.object(sys, "argv", [*argv[:-1], str(output/"checkpoint_006004")]):
                train_main()
            self.assertEqual(updates[-1], 6004)


if __name__ == "__main__":
    unittest.main()
