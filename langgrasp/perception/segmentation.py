"""Instance segmentation wrapper with three interchangeable backends.

- "pt": ultralytics YOLO(.pt) in PyTorch (fp32, or fp16 with half=True).
- "onnx": onnxruntime on the static 640x640 ONNX export (opset 17, no NMS in the graph). Preprocessing
  (letterbox), NMS and prototype-mask decoding are implemented here in numpy so the graph stays plugin free.
- "engine": ultralytics TensorRT engine through YOLO(path) (the engine input is static 1x3x640x640).

detect() always returns full-resolution boolean masks in the coordinates of the input image.
"""

from __future__ import annotations

import ast
import os
import time

import cv2
import numpy as np

from langgrasp.perception.classes import CLASS_NAMES

_EXT_TO_BACKEND = {".pt": "pt", ".onnx": "onnx", ".engine": "engine"}


def letterbox(img: np.ndarray, new_hw: tuple[int, int], color: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with unchanged aspect ratio and pad to new_hw. Returns (image, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    nh, nw = new_hw
    r = min(nh / h, nw / w)
    rw, rh = int(round(w * r)), int(round(h * r))
    if (rw, rh) != (w, h):
        img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    px, py = (nw - rw) // 2, (nh - rh) // 2
    out = np.full((nh, nw, 3), color, dtype=np.uint8)
    out[py : py + rh, px : px + rw] = img
    return out, r, (px, py)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS on (N,4) xyxy boxes. Returns kept indices sorted by descending score."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


class Segmenter:
    """YOLO11-seg wrapper. detect(rgb) -> list of dict(cls, name, conf, box [x1,y1,x2,y2], mask HxW bool)."""

    def __init__(
        self,
        path: str,
        backend: str | None = None,
        device: str = "cuda:0",
        half: bool = False,
        imgsz: int = 640,
        iou: float = 0.5,
        names: list[str] | None = None,
    ):
        self.path = path
        self.backend = backend or _EXT_TO_BACKEND[os.path.splitext(path)[1]]
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self.iou = iou
        self.names = list(names) if names else None
        self.last_timing: dict[str, float] = {}
        if self.backend in ("pt", "engine"):
            from ultralytics import YOLO

            self.model = YOLO(path, task="segment")
            if self.names is None:
                n = getattr(self.model, "names", None)
                self.names = [n[i] for i in sorted(n)] if isinstance(n, dict) else (list(n) if n else list(CLASS_NAMES))
        elif self.backend == "onnx":
            import onnxruntime as ort

            if device.startswith("cuda") and hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()  # find the pip-installed CUDA 12 / cuDNN 9 libraries (nvidia-* wheels)
                except Exception:  # noqa: BLE001
                    pass
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
            so = ort.SessionOptions()
            so.log_severity_level = 3
            self.session = ort.InferenceSession(path, so, providers=providers)
            self.providers = self.session.get_providers()
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
            self.output_names = [o.name for o in self.session.get_outputs()]
            meta = self.session.get_modelmeta().custom_metadata_map
            if self.names is None:
                if "names" in meta:
                    d = ast.literal_eval(meta["names"])
                    self.names = [d[i] for i in sorted(d)]
                else:
                    self.names = list(CLASS_NAMES)
            self.nc = len(self.names)
        else:
            raise ValueError(f"unknown backend {self.backend}")

    # ------------------------------------------------------------------ public API
    def warmup(self, n: int = 3, shape: tuple[int, int] = (480, 640)):
        img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        for _ in range(n):
            self.detect(img)

    def detect(self, rgb: np.ndarray, conf: float = 0.25) -> list[dict]:
        if self.backend == "onnx":
            return self._detect_onnx(rgb, conf)
        return self._detect_ultralytics(rgb, conf)

    # ------------------------------------------------------------------ ultralytics backends
    def _detect_ultralytics(self, rgb: np.ndarray, conf: float) -> list[dict]:
        t0 = time.perf_counter()
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # ultralytics treats numpy input as BGR
        kw = {"quantize": 16} if (self.half and self.backend == "pt") else {}
        res = self.model.predict(
            bgr,
            imgsz=self.imgsz,
            conf=conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            retina_masks=True,
            **kw,
        )[0]
        h, w = rgb.shape[:2]
        out = []
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            masks = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None else None
            for i in range(len(boxes)):
                m = masks[i] if masks is not None else np.zeros((h, w), dtype=bool)
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                out.append({"cls": int(cls[i]), "name": self.names[int(cls[i])], "conf": float(confs[i]), "box": boxes[i].tolist(), "mask": m})
        total = (time.perf_counter() - t0) * 1000
        sp = res.speed
        self.last_timing = {"preprocess_ms": sp["preprocess"], "inference_ms": sp["inference"], "postprocess_ms": sp["postprocess"], "total_ms": total}
        return out

    # ------------------------------------------------------------------ onnx backend
    def _preprocess(self, rgb: np.ndarray):
        lb, r, (px, py) = letterbox(rgb, self.input_hw)
        x = lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), r, (px, py)

    def _detect_onnx(self, rgb: np.ndarray, conf: float) -> list[dict]:
        t0 = time.perf_counter()
        x, r, (px, py) = self._preprocess(rgb)
        t1 = time.perf_counter()
        pred, proto = self.session.run(self.output_names, {self.input_name: x})
        t2 = time.perf_counter()
        out = self._postprocess(pred[0], proto[0], rgb.shape[:2], r, (px, py), conf)
        t3 = time.perf_counter()
        self.last_timing = {
            "preprocess_ms": (t1 - t0) * 1000,
            "inference_ms": (t2 - t1) * 1000,
            "postprocess_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }
        return out

    def _postprocess(self, pred: np.ndarray, proto: np.ndarray, orig_hw: tuple[int, int], r: float, pad: tuple[int, int], conf: float) -> list[dict]:
        """pred: (4+nc+32, N) with xywh boxes in letterboxed pixels; proto: (32, H/4, W/4)."""
        nc = self.nc
        p = pred.T  # (N, 4+nc+32)
        scores = p[:, 4 : 4 + nc]
        cls = scores.argmax(1)
        confs = scores[np.arange(len(p)), cls]
        keep = confs >= conf
        p, cls, confs = p[keep], cls[keep], confs[keep]
        if len(p) == 0:
            return []
        xywh = p[:, :4]
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        # class-aware NMS by offsetting boxes per class
        off = cls[:, None] * float(max(self.input_hw))
        idx = nms_xyxy(boxes + off, confs, self.iou)[:300]
        boxes, cls, confs, coef = boxes[idx], cls[idx], confs[idx], p[idx, 4 + nc :]
        h, w = orig_hw
        px, py = pad
        ih, iw = self.input_hw
        mh, mw = proto.shape[1:]
        masks = coef @ proto.reshape(32, -1)  # (n, mh*mw)
        masks = 1.0 / (1.0 + np.exp(-masks))
        masks = masks.reshape(-1, mh, mw)
        sx, sy = mw / iw, mh / ih
        out = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            m = masks[i]
            # crop the low-res mask to its box (as ultralytics does), then upsample and unpad
            bx1, by1 = max(0, int(np.floor(x1 * sx))), max(0, int(np.floor(y1 * sy)))
            bx2, by2 = min(mw, int(np.ceil(x2 * sx))), min(mh, int(np.ceil(y2 * sy)))
            crop = np.zeros_like(m)
            crop[by1:by2, bx1:bx2] = m[by1:by2, bx1:bx2]
            up = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_LINEAR)
            up = up[py : ih - py if py else ih, px : iw - px if px else iw]
            full = cv2.resize(up, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            box = np.array([(x1 - px) / r, (y1 - py) / r, (x2 - px) / r, (y2 - py) / r])
            box = np.clip(box, 0, [w, h, w, h])
            out.append({"cls": int(cls[i]), "name": self.names[int(cls[i])], "conf": float(confs[i]), "box": box.tolist(), "mask": full})
        return out
