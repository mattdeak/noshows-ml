from runner import Runner
from analyzer import Analyzer
import numpy as np
from plotting_utils import aggregate_reports


if __name__ == '__main__':
    intention_runner = Runner('intention','specs/intention_spec.json')
    pulsar_runner = Runner('pulsar','specs/pulsar_spec.json')

    intention_analyzer = Analyzer('intention')
    pulsar_analyzer = Analyzer('pulsar')

    print("Running Customer Intention Experiments")
    intention_runner.run()

    print("Running Pulsar Detection Experiments")
    pulsar_runner.run()

    print("Generating Plots")
    intention_analyzer.generate_all_plots()
    pulsar_analyzer.generate_all_plots()

    print('--------')
    print("Test Results: Intention")
    print(aggregate_reports("intention"))

    print()
    print("Test Results: Pulsar")

    print(aggregate_reports("pulsar"))

