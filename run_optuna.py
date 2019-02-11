import optuna
import subprocess
import os
import sys
import json


def run_train(shell_command, working_dir=None, result_dir=None, env=None):
    proc = subprocess.Popen(shell_command.split(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=working_dir, env=env)

    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, 'log.out'), 'w') as log_file:
            for line in proc.stdout:
                decoded_line = line.decode('utf-8')
                sys.stdout.write(decoded_line)
                log_file.write(decoded_line)
    proc.wait()
    # if True:
    if proc.returncode != 0:
        raise RuntimeError(
            'invalid return code. Return code:{}'.format(proc.returncode))

def objective(trial):
    working_dir ='./'
    num_iterator_workers = trial.suggest_int('NUM_ITERATOR_WORKERS', 1, 16)
    num_openmp_threads = trial.suggest_int('NUMPY_NUM_THREADS', 1, 16)

    out = "optuna_results/{study_id}-{trial_id}".format(study_id=trial.study_id, trial_id=trial.trial_id)

    shell_command = 'mpirun -n 8 -bind-to none -- '
    shell_command += './task_wrapper.sh {out} {num_iterator_workers} {num_openmp_threads}'.format(out=out,
                                                                                                 num_iterator_workers=num_iterator_workers,
                                                                                                 num_openmp_threads=num_openmp_threads,)

    print('shell_command:', shell_command)
    run_train(shell_command, working_dir, result_dir=out)
    chainer_log_file = os.path.join(out, 'log.json')
    with open(chainer_log_file) as f:
        chainer_log = json.load(f)
    return chainer_log[-1]['elapsed_time']

study = optuna.create_study(study_name='chainermn_cpu', storage='sqlite:///example.db', load_if_exists=True)
study.optimize(objective, n_trials=1)

print(study.best_params)
print(study.best_value)
