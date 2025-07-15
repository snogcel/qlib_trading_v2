from qlib.rl.order_execution.state import SAOEState

class SAOEExtendedState(SAOEState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_bin = None  # or bin_selected
