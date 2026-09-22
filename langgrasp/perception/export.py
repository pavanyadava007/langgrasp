"""Export the fine-tuned YOLO11n-seg to ONNX (opset 17, static 1x3x640x640, simplified, no NMS in the graph) and
to TensorRT FP16 and INT8 engines (built on the local GPU; TensorRT engines are not portable across GPU types)."""

from __future__ import annotations

import os
import shutil
import time

IMGSZ = 640
OPSET = 17


def _stem(pt_path: str) -> str:
    return os.path.splitext(pt_path)[0]


def export_onnx(pt_path: str) -> str:
    from ultralytics import YOLO

    out = YOLO(pt_path).export(format="onnx", imgsz=IMGSZ, opset=OPSET, simplify=True, dynamic=False, batch=1, nms=False)
    return str(out)


def export_engine(pt_path: str, precision: str, data_yaml: str | None = None, workspace_gb: float = 2.0) -> str:
    """precision: 'fp16' or 'int8' (INT8 needs data_yaml for calibration). Returns the renamed engine path."""
    from ultralytics import YOLO

    kw = {"quantize": 16} if precision == "fp16" else {"quantize": 8, "data": data_yaml}
    out = YOLO(pt_path).export(format="engine", imgsz=IMGSZ, opset=OPSET, simplify=True, dynamic=False, batch=1, workspace=workspace_gb, nms=False, **kw)
    dst = f"{_stem(pt_path)}-{precision}.engine"
    shutil.move(str(out), dst)
    return dst


def check_onnx(onnx_path: str) -> dict:
    """Verify static shape, opset and that only the standard ONNX domain is used."""
    import onnx

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    inp = m.graph.input[0]
    dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
    opsets = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    custom = sorted({n.domain for n in m.graph.node if n.domain not in ("", "ai.onnx")})
    outs = {o.name: [d.dim_value if d.dim_value > 0 else d.dim_param for d in o.type.tensor_type.shape.dim] for o in m.graph.output}
    return {
        "input": inp.name,
        "input_shape": dims,
        "static": all(isinstance(d, int) for d in dims),
        "opsets": opsets,
        "custom_domains": custom,
        "outputs": outs,
        "n_nodes": len(m.graph.node),
        "size_mb": round(os.path.getsize(onnx_path) / 1e6, 2),
    }


def export_all(pt_path: str, data_yaml: str, do_int8: bool = True) -> dict:
    """Engines first (their intermediate ONNX shares the .onnx filename), then the final ONNX. Returns paths + timings."""
    info: dict = {"weights": pt_path, "imgsz": IMGSZ, "opset": OPSET}
    t = time.time()
    info["engine_fp16"] = export_engine(pt_path, "fp16")
    info["engine_fp16_build_s"] = round(time.time() - t, 1)
    if do_int8:
        t = time.time()
        info["engine_int8"] = export_engine(pt_path, "int8", data_yaml)
        info["engine_int8_build_s"] = round(time.time() - t, 1)
    t = time.time()
    info["onnx"] = export_onnx(pt_path)
    info["onnx_export_s"] = round(time.time() - t, 1)
    info["onnx_check"] = check_onnx(info["onnx"])
    return info
