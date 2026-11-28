import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from neural_rx.receiver.training_options import requested_updates
from neural_rx.cli.train import main as train_main
from neural_rx.receiver.checkpoints import recover_history


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

    def test_training_log_recovers_only_updates_in_the_checkpoint(self):
        def row(step):
            return json.dumps(dict(update=step,width=12,episode_fingerprints=[str(step)])).encode()
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
                    "neural_rx.receiver.training":SimpleNamespace(MemoryTrainer=Trainer,atomic_json=write_json)}):
            source = publish_checkpoint(Path(temporary)/"source",6000)
            output = Path(temporary)/"continued"
            for resume in (source,output/"checkpoint_006000"):
                with mock.patch.object(sys,"argv",["train","--output-dir",str(output),
                                                   "--data-gpu","none","--replica-gpu","none",
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
                mock.patch.dict(sys.modules,{"tensorflow":tensorflow,
                                             "neural_rx.receiver.training":training_module}), \
                mock.patch("builtins.print"):
            output = Path(temporary)
            checkpoint = publish_checkpoint(output,5999)
            for stop in (6001, 6003):
                with mock.patch.object(sys,"argv",["train","--output-dir",str(output),
                                                   "--data-gpu","none","--replica-gpu","none",
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

    def test_development_results_reach_the_schedule_a_fixed_number_of_updates_later(self):
        from neural_rx.cli.train import parser, run_training

        class Trainer:
            def __init__(self, *args):
                self.value, self.seed, self.seen = 0, 1, -1
                self.episodes = SimpleNamespace()
                self.step = SimpleNamespace(numpy=lambda: self.value)

            def binding(self):
                return {}

            def schedule_state(self):
                return {"development_update": self.seen}

            def restore(self, path):
                self.value = json.loads((Path(path)/"state.json").read_text())["update"]
                self.seen = seen[self.value]

            def save(self, path):
                seen[self.value] = self.seen
                return publish_checkpoint(path, self.value)

            def update(self, packets):
                self.value += 1
                return {"update": self.value}, []

            def observe_development(self, result):
                self.seen = result["update"]
                observed.append((result["update"], self.value))

        def evaluate(checkpoint):
            return dict(checkpoint=str(checkpoint), update=int(Path(checkpoint).name.split("_")[1]))

        tensorflow = SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda _: [],
                                    experimental=SimpleNamespace(enable_op_determinism=lambda: None)))
        training_module = SimpleNamespace(MemoryTrainer=Trainer, atomic_json=write_json)
        recipe = ROOT/"config/memory_training.json"
        for delay, resume_at, expected in ((0, None, [(4, 4), (8, 8)]), (2, None, [(4, 6), (8, 10)]),
                                           (2, 5, [(4, 6), (8, 10)]), (3, None, [(4, 7), (8, 10)])):
            observed, seen = [], {}       # seen: the schedule's last development update, per saved update
            with self.subTest(delay=delay, resume_at=resume_at), tempfile.TemporaryDirectory() as temporary, \
                    mock.patch.dict(os.environ), mock.patch("builtins.print"), \
                    mock.patch.dict(sys.modules, {"tensorflow": tensorflow,
                                                  "neural_rx.receiver.training": training_module}):
                output = Path(temporary)
                argv = ["--recipe", str(recipe), "--output-dir", str(output), "--data-gpu", "none",
                        "--replica-gpu", "none", "--steps", "10"]
                if resume_at is not None:
                    # a run that stopped between the development update 4 and the use of its result
                    publish_checkpoint(output, 4)
                    checkpoint = publish_checkpoint(output, resume_at)
                    seen[resume_at] = -1
                    argv += ["--resume", str(checkpoint)]
                run_training(parser().parse_args(argv), evaluation_points=[4, 8], evaluate=evaluate,
                             evaluation_delay=delay)
                self.assertEqual(observed, expected)
        with self.assertRaisesRegex(ValueError, "before the next development update"):
            run_training(parser().parse_args(["--recipe", str(recipe), "--output-dir", "unused", "--steps", "10"]),
                         evaluation_points=[4, 8], evaluate=evaluate, evaluation_delay=4)


if __name__ == "__main__":
    unittest.main()
