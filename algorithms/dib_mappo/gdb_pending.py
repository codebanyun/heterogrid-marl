"""Executed only by GDB's Python interpreter, not by the training interpreter."""
import os
from pathlib import Path

import gdb

try:
    payload = Path(os.environ["DIB_PENDING_PAYLOAD"]).resolve()
    if any(char.isspace() for char in str(payload)):
        raise RuntimeError("Unexpected whitespace in payload path")
    size = (payload.stat().st_size // 4096 + 1) * 4096
    gdb.execute("set pagination off")
    gdb.execute("set $dib_upgrade_memory = (void *) mmap((void *) 0, %d, 3, 34, -1, 0)" % size)
    address = int(gdb.parse_and_eval("(unsigned long) $dib_upgrade_memory"))
    if address in (0, 2**64 - 1):
        raise RuntimeError("mmap failed")
    # mmap supplies the trailing zero terminator; restore does not call into
    # the inferior Python allocator or require the inferior to hold the GIL.
    gdb.execute("restore %s binary $dib_upgrade_memory" % payload)
    status = int(gdb.parse_and_eval(
        "(int) Py_AddPendingCall((int (*)(void *)) PyRun_SimpleString, $dib_upgrade_memory)"))
    if status != 0:
        raise RuntimeError("CPython pending-call queue rejected the callback")
    print("DIB_PENDING_SCHEDULED=0")
finally:
    gdb.execute("detach")
