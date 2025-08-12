# logging_relay.py

from asyncio import Queue

class SmartQALogRelay:
    def __init__(self):
        self.queues: list[Queue] = []

    def register(self):
        q = Queue()
        self.queues.append(q)
        return q

    def unregister(self, q):
        if q in self.queues:
            self.queues.remove(q)

    def log(self, msg: str):
        # Push to websocket queues only; avoid terminal prints to prioritize UI streaming
        for q in self.queues:
            q.put_nowait(msg)

    def clear(self):
        self.queues.clear()

# Singleton relay for global use
smartqa_log_relay = SmartQALogRelay()

def log(msg: str):
    smartqa_log_relay.log(msg)
