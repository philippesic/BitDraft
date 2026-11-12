import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    boot: str
    birth: str

    @classmethod
    def read(cls, pid: int) -> "ProcessIdentity":
        text = Path(f"/proc/{pid}/stat").read_text()
        tail = text[text.rfind(")") + 2 :].split()
        return cls(pid, Path("/proc/sys/kernel/random/boot_id").read_text().strip(), tail[19])

    def alive(self) -> bool:
        try:
            text = Path(f"/proc/{self.pid}/stat").read_text()
            tail = text[text.rfind(")") + 2 :].split()
            return (
                tail[0] not in {"Z", "X"}
                and tail[19] == self.birth
                and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == self.boot
            )
        except (OSError, IndexError):
            return False

    def signal_group(self, value: int) -> None:
        if not self.alive() or os.getpgid(self.pid) != self.pid:
            raise ValueError("owned process birth or isolated group no longer matches")
        os.killpg(self.pid, value)
