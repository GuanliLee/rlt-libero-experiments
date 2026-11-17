"""Reuse the original LIBERO subprocess without changing reset/step semantics.

The pinned worker still constructs and seeds a fresh simulator on every reset.
Only Python imports and subprocess creation are amortized across episodes.
"""
from pathlib import Path
import threading


class PersistentEpisodeEnv:
    def __init__(self, env, stderr_log):
        self._env = env
        self._finalized = False
        stderr_log = Path(stderr_log)
        stderr_log.parent.mkdir(parents=True, exist_ok=True)
        self._log = stderr_log.open('ab')
        self._drain_error = None
        self._thread = threading.Thread(target=self._drain_stderr, name='libero-stderr', daemon=True)
        self._thread.start()

    def _drain_stderr(self):
        try:
            while True:
                data = self._env._proc.stderr.read1(8192)
                if not data:
                    break
                self._log.write(data)
                self._log.flush()
        except (OSError, ValueError) as exc:
            self._drain_error = repr(exc)
        finally:
            self._log.close()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        """Episode cleanup keeps the process alive; shutdown owns final cleanup."""

    def shutdown(self):
        if self._finalized:
            return
        self._finalized = True
        self._env.close()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError('LIBERO stderr drainer did not finish after worker shutdown')
        if self._drain_error:
            raise RuntimeError(f'LIBERO stderr draining failed: {self._drain_error}')


def install_persistent_env(exp, stderr_log=None):
    """Replace exp.Env with a lazy single-worker factory; return final cleanup.

    The caller must call the returned function from a finally block. Existing
    Experiment.episode remains unchanged, including its per-episode close().
    """
    original_factory = exp.Env
    stderr_log = Path(stderr_log) if stderr_log else Path(exp.output) / 'libero-worker-stderr.log'
    holder = []

    def factory(*args, **kwargs):
        if args or kwargs:
            raise TypeError('Persistent episode factory only supports the no-argument Env() API')
        if not holder:
            holder.append(PersistentEpisodeEnv(original_factory(), stderr_log))
        return holder[0]

    def close_final():
        exp.Env = original_factory
        if holder:
            holder[0].shutdown()

    exp.Env = factory
    return close_final
