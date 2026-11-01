import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


RUN = Path(__file__).resolve().parent
SOURCE = RUN / 'source'
sys.path.insert(0, str(SOURCE / 'scripts'))
from artifact_io import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    config = json.loads((RUN / 'config.json').read_text())
    provenance = json.loads((RUN / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        if file_sha256(RUN / name) != expected:
            raise ValueError('Recorded source or input changed: ' + name)
    for bank in provenance['evaluation_banks'].values():
        if file_sha256(Path(bank['path']) / 'dataset.json') != bank['manifest_sha256']:
            raise ValueError('Evaluation dataset changed: ' + bank['path'])
    environment = dict(os.environ, PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='2')
    environment['PYTHONPATH'] = os.pathsep.join((str(SOURCE / 'scripts'), str(SOURCE)))
    command = list(config['campaign_command'])
    if args.dry_run:
        subprocess.run(command + ['--dry-run'], cwd=SOURCE, env=environment, check=True)
        return
    if args.initialize:
        command[command.index('--steps') + 1] = '0'
    status_path = RUN / ('initialization_status.json' if args.initialize else 'status.json')
    status = dict(pid=os.getpid(), requested_updates=0 if args.initialize else config['requested_updates'],
                  started=datetime.now(timezone.utc).isoformat(), status='running')
    atomic_json(status_path, status)
    try:
        print(json.dumps(dict(command=command, cwd=str(SOURCE))), flush=True)
        subprocess.run(command, cwd=SOURCE, env=environment, check=True)
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['finished'] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, status)


if __name__ == '__main__':
    main()
