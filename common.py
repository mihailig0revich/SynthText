import sys
import signal
from contextlib import contextmanager

class Color: #pylint: disable=W0232
    GRAY=30
    RED=31
    GREEN=32
    YELLOW=33
    BLUE=34
    MAGENTA=35
    CYAN=36
    WHITE=37
    CRIMSON=38    

def colorize(num, string, bold=False, highlight = False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def colorprint(colorcode, text, o=sys.stdout, bold=False):
    o.write(colorize(colorcode, text, bold=bold))

def warn(msg):
    print (colorize(Color.YELLOW, msg))

def error(msg):
    print (colorize(Color.RED, msg))

# http://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Cross-platform time limit.
    POSIX: uses SIGALRM/ITIMER_REAL; Windows: no-op (no SIGALRM)."""
    if hasattr(signal, "SIGALRM"):
        # POSIX путь
        def _handler(signum, frame):
            raise TimeoutException(colorize(Color.RED, "   *** Timed out!", highlight=True))

        old_handler = signal.signal(signal.SIGALRM, _handler)
        try:
            # Если доступен setitimer — используем более точный таймер
            if hasattr(signal, "setitimer") and hasattr(signal, "ITIMER_REAL"):
                signal.setitimer(signal.ITIMER_REAL, seconds)
            else:
                signal.alarm(seconds)
            yield
        finally:
            # Сбрасываем таймеры и хэндлер
            if hasattr(signal, "setitimer") and hasattr(signal, "ITIMER_REAL"):
                signal.setitimer(signal.ITIMER_REAL, 0)
            else:
                signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows: таймаут не поддерживается этим методом
        # (можно сделать строгий таймаут через multiprocessing, но это уже рефакторинг)
        yield
