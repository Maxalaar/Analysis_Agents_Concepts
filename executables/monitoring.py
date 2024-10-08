import os
import time
import webbrowser
from multiprocessing import Process
from typing import List


def run_tensorboard():
    path = os.getcwd()
    path = os.path.dirname(path)
    print(path)
    os.system('tensorboard --bind_all --logdir ' + str(path) + '/experiments/')


def list_to_url(list_monitoring: List[str]) -> str:
    url: str = ''
    url += 'http://localhost:6006/?pinnedCards=['

    for element in list_monitoring:
        url += '{"plugin"%3A"scalars"%2C"tag"%3A"'
        url += element.replace('/', '%2F')
        url += '"}'
        url += '%2C'

    url = url[:-3]
    url += ']&darkMode=true#timeseries'

    url = url.replace('[', '%5B')
    url = url.replace(']', '%5D')
    url = url.replace('{', '%7B')
    url = url.replace('}', '%7D')
    url = url.replace('"', '%22')

    return url


if __name__ == "__main__":
    list_learning = [
        'ray/tune/evaluation/env_runners/episode_reward_mean',
        'ray/tune/env_runners/episode_reward_mean',
        'ray/tune/info/learner/default_policy/learner_stats/policy_loss',
        'ray/tune/info/learner/default_policy/learner_stats/vf_loss',
        'ray/tune/info/learner/default_policy/learner_stats/entropy',
        'ray/tune/info/learner/default_policy/model/kullback_leibler_loss',
        'ray/tune/info/learner/default_policy/model/policy_loss',
    ]

    list_simulation = [
        'ray/tune/sampler_perf/mean_env_render_ms',
        'ray/tune/sampler_perf/mean_action_processing_ms',
        'ray/tune/sampler_perf/mean_inference_ms',
        'ray/tune/perf/cpu_util_percent',
        'ray/tune/perf/gpu_util_percent0',
        'ray/tune/perf/ram_util_percent',
    ]

    list_supervised = [
        'supervised/-train_loss',
        'supervised/-validation_loss',
    ]

    list_perturbation = [
        'perturbation/-train_loss',
        'perturbation/-train_loss_classification',
        'perturbation/-train_loss_regression',
        'perturbation/-validation_loss',
        'perturbation/-validation_loss_classification',
        'perturbation/-validation_loss_regression',
    ]

    process: Process = Process(target=run_tensorboard)
    process.start()
    time.sleep(5)

    # Monitoring Ray Dashboard
    webbrowser.open_new('127.0.0.1:8265')

    # Monitoring simulation
    webbrowser.open(list_to_url(list_simulation))

    # Monitoring of learning
    webbrowser.open(list_to_url(list_learning))

    # Monitoring of supervised
    webbrowser.open(list_to_url(list_supervised))

    # Monitoring of perturbation
    webbrowser.open(list_to_url(list_perturbation))
