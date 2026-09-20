import subprocess
import sys
import unittest

import processmanager


class ProcessEnvironmentTests(unittest.TestCase):
    def test_child_receives_requested_environment(self):
        options = processmanager.subprocess_args(False, {"WT_CHILD_PROBE": "linux-runtime"})
        output = subprocess.check_output(
            [sys.executable, "-c", "import os; print(os.environ['WT_CHILD_PROBE'])"],
            **options,
        )
        self.assertEqual(output.strip(), b"linux-runtime")
