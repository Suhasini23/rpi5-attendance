import time

class Debounce:
    def __init__(self, cooldown_sec=120):
        self.cooldown = cooldown_sec
        self.last = {}

    def ok(self, name):
        now = time.time()
        t = self.last.get(name, 0)
        if now - t >= self.cooldown:
            self.last[name] = now
            return True
        return False
