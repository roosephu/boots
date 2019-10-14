#!/usr/bin/env python
from itertools import product
import argparse
import datetime
import re
import sys
from multiprocessing import Pool
from subprocess import run, DEVNULL
from termcolor import colored, cprint
import toml
from pathlib import Path


def replace(s, a, b):
    return re.sub(f'\\b{a}\\b', b, s)


def execute(template, assignment, stdout, stderr):
    cmd = template
    for key, value in assignment:
        cmd = replace(cmd, key, value)
    print(f'{cmd}')
    run(cmd, shell=True, stdin=DEVNULL, stdout=stdout, stderr=stderr, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', help='template', required=True)
    parser.add_argument('--map', help='replace arguments', nargs=2, action='append', metavar=('PATH', 'VALUE'),
                        default=[])
    parser.add_argument('--log_dir', help='the directory to logs', default='/tmp')
    parser.add_argument('--n_jobs', '-n', help='# jobs per assignment', default=0, type=int)
    parser.add_argument('--pool_size', '-p', help='size of process pool, default to # total jobs', default=0, type=int)
    parser.add_argument('--stdout', help='whether to use stdout', default=False, action='store_true')
    parser.add_argument('--stderr', help='whether to use stderr', default=False, action='store_true')

    args = parser.parse_args()
    uid = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    log_dir = replace(args.log_dir, 'UID', uid)
    cprint(f'log dir = {log_dir}', 'red', attrs=['bold', 'underline'])
    stdout = None if args.stdout else DEVNULL
    stderr = None if args.stderr else DEVNULL
    assignments = list(product([['LOGDIR', log_dir]], *[[[k, v] for v in values.split(',')] for k, values in args.map]))

    log_dir = Path(log_dir).expanduser()
    log_dir.mkdir(exist_ok=False, parents=True)
    with open(log_dir / '.status.toml', 'w') as f:
        toml.dump({'cmd': sys.argv}, f)

    pool_size = len(assignments) * args.n_jobs
    if args.pool_size != 0:
        pool_size = min(args.pool_size, pool_size)
    pool = Pool(processes=pool_size)
    pool.starmap(execute, [[args.template, assignment, stdout, stderr] for assignment in assignments * args.n_jobs])
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
