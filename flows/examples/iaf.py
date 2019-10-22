from flows.examples.utils import run_example
from flows import IAF


class FlowLayer(IAF):
    def __init__(self, size):
        super().__init__(size, auto_regressive_hidden=16)


run_example(FlowLayer, n_flows=4, target_function_name='u3')

