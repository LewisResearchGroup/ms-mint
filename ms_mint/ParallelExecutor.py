import time
from multiprocessing import Pool, Manager, cpu_count


class ParallelExecutor:
    def __init__(self, function, args, processes=None, progress_callback=None):
        self.args = args
        self.function = function
        self._progress = 0
        self.progress_callback = progress_callback

        if processes is None:
            self.processes = cpu_count()
        else:
            self.processes = processes

    def run(self):
        pool = Pool(processes=self.processes)
        m = Manager()
        q = m.Queue()
        args = self.args

        results = pool.map_async(self.function, args)

        while True:
            if results.ready():
                break
            else:
                size = q.qsize()
                progress = int(100 * (size + 1) / len(args))
                self.progress = progress
                time.sleep(1)

        self.progress = 100

        pool.close()
        pool.join()
        return results.get()

    @property
    def progress(self):
        return self.progress

    @progress.setter
    def progress(self, value):
        self._progress = value
        if self.progress_callback is not None:
            self.progress_callback(value)
