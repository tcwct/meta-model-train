from __future__ import annotations

import json
import platform
import sys


def main() -> None:
    payload: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostic path
        payload["torch_import_ok"] = False
        payload["torch_error"] = repr(exc)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    payload["torch_import_ok"] = True
    payload["torch_version"] = torch.__version__
    payload["cuda_runtime"] = torch.version.cuda
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["gpu_count"] = int(torch.cuda.device_count())
    payload["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
