import datetime
import re
import itertools
import os

import lunzi as lz
from lunzi.typing import *


class FLAGS(lz.BaseFLAGS):
    template = ''
    name = ''
    n_jobs = 0
    replace = []

    @classmethod
    def finalize(cls):
        assert cls.name


@lz.main(FLAGS)
@FLAGS.inject
def main(*, name, template, replace, n_jobs, _log: Logger):
    run_id = f'{name}-{datetime.datetime.now().strftime("%y%m%d-%H%M%S")}'
    keys = ['ID']
    values = [[run_id]]
    for r in replace:
        key, value = r.split('=', 2)
        keys.append(re.compile(f'\\b{key}\\b'))
        values.append(value.split(','))
    values = itertools.product(*values)

    for assignment in values:
        cmd = template
        for key, value in zip(keys, assignment):
            cmd = key.sub(cmd, value)
        _log.info(f'cmd')
        for _ in range(n_jobs):
            os.system(cmd)


if __name__ == '__main__':
    main()
