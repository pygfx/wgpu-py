import atexit
import threading


is_shutting_down = False


@atexit.register
def mark_shutdown():
    global is_shutting_down
    is_shutting_down = True


class PollToken:
    """A token for the poller to be obtained via PollThread.get_token().

    For as long as the token is active (alive and set_done() is not called),
    the poll thread will keep polling.
    """

    def __init__(self, id, ids):
        self._id = id
        self._ids = ids

    def set_done(self):
        """Mark the token as done to relief the poll thread of its polling work for this token."""
        self._ids.discard(self._id)

    def is_done(self):
        return self._id not in self._ids

    def __del__(self):
        self.set_done()


class PollThread(threading.Thread):
    """A thread to poll the device, but only when needed.

    The thread actively poll when there is stuff waiting. Compared to polling
    periodically, this results in less battery drain, while having faster
    responses. Inspired by the implementation wgpu-polling in the Servo browser.

    Relevant links:

    * https://github.com/sagudev/servo/blob/main/components/webgpu/poll_thread.rs
    * https://github.com/servo/servo/pull/32266
    * https://bugzilla.mozilla.org/show_bug.cgi?id=1870699

    """

    def __init__(self, poll_func):
        super().__init__()
        self._poll_func = poll_func
        self._token_ids = set()  # note that add and discard are atomic under the GIL
        self._token_count = 0
        self._token_id_lock = threading.Lock()
        self._event = threading.Event()
        self._shutdown = False
        self.daemon = True  # don't let this thread prevent shutdown

    def get_token(self):
        """Awake the poll thread and get a PollToken.

        The thread will keep polling the device until the token's ``set_done()``
        method is called, or it is deleted by the garbage collector.
        """
        if self._shutdown:
            raise RuntimeError("Cannot use PollThread because it has stopped.")

        with self._token_id_lock:
            self._token_count += 1
            token_id = self._token_count

        # First add the token id to the set, then wake the thread. The thread
        # will now keep on polling (with block) until the token is done (removed
        # from token_ids).
        self._token_ids.add(token_id)
        token = PollToken(token_id, self._token_ids)
        self._event.set()

        return token

    def stop(self):
        self._shutdown = True
        self._poll_func = lambda _: None
        self._token_ids.clear()
        self._event.set()
        # Python 3.13 can hang when joining this when shutting down. Python 3.14 doesn't even allow it.
        if not is_shutting_down:
            self.join(timeout=1)

    def run(self):
        """The thread logic."""
        # No sleeps, just block waiting for the GPU to finish something, or waiting for a token to be created.

        event = self._event
        token_ids = self._token_ids

        while not self._shutdown:
            # Wait for token to be created
            event.wait()  # blocking
            event.clear()
            # Do one non-blocking call
            self._poll_func(False)
            # Keep polling until the tokens are all done
            while token_ids:
                self._poll_func(True)  # blocking
