"""An advisory lock prevents concurrent writers to one training directory."""
import fcntl
import os


class RunLock:
    def __init__(self, folder):
        self.file = (folder / ".trainer.lock").open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError(f"Another trainer is already using {folder}") from None
        self.file.seek(0)
        self.file.truncate()
        self.file.write(str(os.getpid()) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
