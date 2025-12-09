### embedding_model.py

import logging
import time
from queue import Empty, Queue
from threading import Event, Thread

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")
embedding_queue = Queue()

class EmbeddingTask:
    def __init__(self, text, done: Event, timeout=10):
        self.text = text
        self.done = done
        self.result = None
        self.timeout = timeout

    def wait(self):
        start = time.time()
        while not self.done.is_set():
            if time.time() - start > self.timeout:
                logger.warning("Embedding wait 超時，回傳空向量")
                return [0.0] * 512
            time.sleep(0.01)
        return self.result

def embedding_worker():
    while True:
        tasks = []
        try:
            task = embedding_queue.get()
            tasks.append(task)
            time.sleep(0.01)  # 累積更多 batch
            while not embedding_queue.empty():
                tasks.append(embedding_queue.get())
            texts = [t.text for t in tasks]
            vectors = model.encode(texts)
            for task, vec in zip(tasks, vectors):
                task.result = vec.tolist()
                task.done.set()
        except Exception as e:
            logger.exception("Embedding Worker 錯誤")
            for task in tasks:
                task.result = [0.0] * 512
                task.done.set()

Thread(target=embedding_worker, daemon=True).start()

def get_embedding(text: str) -> list:
    done = Event()
    task = EmbeddingTask(text, done)
    embedding_queue.put(task)
    vec = task.wait()
    logger.info("Embedding shape: %s", len(vec))
    assert len(vec) == 512, f"Embedding 維度錯誤！現在長度: {len(vec)}，預期 512"
    return vec
