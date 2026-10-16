from dataclasses import dataclass


@dataclass(frozen=True)
class RingTypes:
    METAL_RING: str = "metal_ring"
    LARGE_METAL_RING: str = "large_metal_ring"
    PLASTIC_RING: str = "plastic_ring"

    def id(self, name: str) -> int:
        id = None
        if name == self.METAL_RING:
            id = 0
        elif name == self.LARGE_METAL_RING:
            id = 1
        elif name == self.PLASTIC_RING:
            id = 2
        return id

    def name(self, id: int) -> str:
        name = None
        if id == 0:
            name = self.METAL_RING
        elif id == 1:
            name = self.LARGE_METAL_RING
        elif id == 2:
            name = self.PLASTIC_RING
        return name


@dataclass(frozen=True)
class Defaults:
    VIDEO_CODER_DUMP_LEVEL: int = 0
    # CONF: float = 0.25
    # IOU: float = 0.7
    # MAX_DET: int = 2
    N_PROCESSES: int = 4


@dataclass(frozen=True)
class VIDEO_FILE_TYPES:
    AVI: str = ".avi"
    MP4: str = ".mp4"
    TS: str = ".ts"
