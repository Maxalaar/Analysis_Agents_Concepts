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
    ]

    list_simulation = [
        'ray/tune/sampler_perf/mean_env_render_ms',
        'ray/tune/sampler_perf/mean_action_processing_ms',
        'ray/tune/sampler_perf/mean_inference_ms',
        'ray/tune/perf/cpu_util_percent',
        'ray/tune/perf/gpu_util_percent0',
        'ray/tune/perf/ram_util_percent',
    ]

    list_lightning = [
        'lightning/-train_loss',
        'lightning/-validation_loss',
    ]

    process: Process = Process(target=run_tensorboard)
    process.start()
    time.sleep(5)

    # Monitoring simulation
    webbrowser.open_new(list_to_url(list_simulation))

    # Monitoring Ray Dashboard
    webbrowser.open('127.0.0.1:8265')

    # Monitoring of lightning
    webbrowser.open(list_to_url(list_lightning))

    # Monitoring of learning
    webbrowser.open(list_to_url(list_learning))
