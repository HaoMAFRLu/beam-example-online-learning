import numpy as np
        
class StepSize():
    def __init__(self, mode: str, PARAMS: dict) -> None:
        self.mode = mode
        self.PARAMS = PARAMS
        self.initialization()

    def _ini_constant(self):
        """
        """
        self.value0 = self.PARAMS['value0']

    def _ini_step(self):
        self.iterations = self.PARAMS["iterations"]
        self.values = self.PARAMS["values"]
        self.value0 = self.PARAMS["value0"]
        self.num_intervals = len(self.iterations)

    def _ini_decreasing(self):
        """
        """
        self.iterations = self.PARAMS["iterations"]
        self.values = self.PARAMS["values"]
        self.value0 = self.PARAMS["value0"]
        self.num_intervals = len(self.iterations)

        self.a = [None]*self.num_intervals
        self.b = [None]*self.num_intervals
        for i in range(len(self.iterations)):
            if i == 0:
                i1 = 1
                value1 = self.value0
            else:
                i1 = self.iterations[i-1]
                value1 = self.values[i-1]
            
            i2 = self.iterations[i]
            value2 = self.values[i]
            self.a[i], self.b[i] = self.fit_exponential(i1, value1, i2, value2)

    def initialization(self) -> None:
        """
        """
        if self.mode == 'constant':
            self._ini_constant()
        elif self.mode == 'step':
            pass
        elif self.mode == 'decreasing':
            pass

    def get_interval(self, array, x):
        for index, element in enumerate(array):
            if element > x:
                return index
        return None  # Return None if no element is greater than x
    
    def fit_exponential(self, i1, sigma1, i2, sigma2):
        # Solve for b
        # y1 = a * exp(b * x1) and y2 = a * exp(b * x2)
        # => b = ln(y2 / y1) / (x2 - x1)
        b = np.log(sigma2/sigma1)/(i2 - i1)
        # Solve for a
        # y1 = a * exp(b * x1)
        # => a = y1 / exp(b * x1)
        a = sigma1 / np.exp(b * i1)
        return a, b

    def get_exponential(self, a, b, iteration):
        return a*np.exp(b*iteration)

    def _get_eta_step(self, nr_iteration: float) -> float:
        idx_interval = self.get_interval(self.iterations, nr_iteration)
        if idx_interval is None:
            return self.values[-1]
        else:
            return self.values[idx_interval]

    def _get_eta_constant(self, nr_iteration: float) -> float:
        return self.value0
    
    def _get_eta_decreasing(self, nr_iteration: float) -> float:
        idx_interval = self.get_interval(self.iterations, nr_iteration)
        if idx_interval is None:
            return self.values[-1]
        else:
            a = self.a[idx_interval]
            b = self.b[idx_interval]
            return self.get_exponential(a, b, nr_iteration)

    def get_eta(self, nr_iteration):
        if self.mode == 'step':
            return self._get_eta_step(nr_iteration)
        elif self.mode == 'constant':
            return self._get_eta_constant(nr_iteration)
        elif self.mode == 'decreasing':
            return self._get_eta_decreasing(nr_iteration)
