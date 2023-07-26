import atexit
import threading
import time
import subprocess
import os

all_processes = []


# Source: https://github.com/pyinstaller/pyinstaller/wiki/Recipe-subprocess
# Create a set of arguments which make a ``subprocess.Popen`` (and
# variants) call work with or without Pyinstaller, ``--noconsole`` or
# not, on Windows and Linux. Typical use::
#
#   subprocess.call(['program_to_run', 'arg_1'], **subprocess_args())
#
# When calling ``check_output``::
#
#   subprocess.check_output(['program_to_run', 'arg_1'],
#                           **subprocess_args(False))
def subprocess_args(include_stdout=True):
    # The following is true only on Windows.
    if hasattr(subprocess, 'STARTUPINFO'):
        # On Windows, subprocess calls will pop up a command window by default
        # when run from Pyinstaller with the ``--noconsole`` option. Avoid this
        # distraction.
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        # Windows doesn't search the path by default. Pass it an environment so
        # it will.
        #env = os.environ
        # modify environment variables
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "UTF-8"
        env["PYTHONLEGACYWINDOWSSTDIO"] = "UTF-8"
        env["PYTHONUTF8"] = "1"
    else:
        si = None
        env = None

    # ``subprocess.check_output`` doesn't allow specifying ``stdout``::
    #
    #   Traceback (most recent call last):
    #     File "test_subprocess.py", line 58, in <module>
    #       **subprocess_args(stdout=None))
    #     File "C:\Python27\lib\subprocess.py", line 567, in check_output
    #       raise ValueError('stdout argument not allowed, it will be overridden.')
    #   ValueError: stdout argument not allowed, it will be overridden.
    #
    # So, add it only if it's needed.
    if include_stdout:
        ret = {'stdout': subprocess.PIPE}
    else:
        ret = {}

    # On Windows, running this from the binary produced by Pyinstaller
    # with the ``--noconsole`` option requires redirecting everything
    # (stdin, stdout, stderr) to avoid an OSError exception
    # "[Error 6] the handle is invalid."
    ret.update({'stdin': subprocess.PIPE,
                'stderr': subprocess.PIPE,
                'startupinfo': si,
                'env': env})
    return ret


def run_process(process_arguments, include_stdout=False):
    # run command line tool with parameters
    try:
        process = subprocess.Popen(process_arguments, **subprocess_args(include_stdout), close_fds=True)
        all_processes.append(process)
        return process

    except subprocess.CalledProcessError as e:
        return None


def kill_process(process):
    timeout_sec = 5
    p_sec = 0
    for second in range(timeout_sec):
        if process.poll() is None:
            time.sleep(1)
            p_sec += 1
    if p_sec >= timeout_sec:
        process.kill()  # supported from python 2.6
    # remove process from list
    all_processes.remove(process)
    print('Process killed')


def cleanup_subprocesses():
    threads = []
    for p in all_processes:  # list of your processes
        kill_thread = threading.Thread(target=kill_process, args=(p,))
        kill_thread.start()
        threads.append(kill_thread)

    # wait for all threads to finish
    for t in threads:
        t.join()

    print('All processes killed')


atexit.register(cleanup_subprocesses)
