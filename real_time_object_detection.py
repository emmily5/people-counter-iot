"""Deteccao de objetos em video com SSD MobileNetV1 (ST ONNX) e OpenCV 5.

Requisitos: pip install numpy opencv-python
Modelo: models/ssd_mobilenetv1_pt_voc_300.onnx (STMicroelectronics, PASCAL VOC)

Exemplos:
    # Webcam padrao (camera 0)
    python real_time_object_detection.py -m models/ssd_mobilenetv1_pt_voc_300.onnx

    # Outra webcam, arquivo ou stream RTSP
    python real_time_object_detection.py -m models/ssd_mobilenetv1_pt_voc_300.onnx -v 1
    python real_time_object_detection.py -m models/ssd_mobilenetv1_pt_voc_300.onnx -v video.mp4
    python real_time_object_detection.py -m models/ssd_mobilenetv1_pt_voc_300.onnx -v 'rtsp://usuario:senha@host:554/stream'

    # Salvar video anotado sem abrir janela
    python real_time_object_detection.py -m models/ssd_mobilenetv1_pt_voc_300.onnx \
        -v video.mp4 --output resultado.mp4 --no-display

Pressione q ou ESC para parar a exibicao. Ctrl+C tambem interrompe a captura.

IMPORTANTE: a configuracao SSD de 3000 ancoras reproduz a versao de imagens
previamente validada visualmente, nao uma parametrizacao geral de SSD.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
INPUT_SIZE = (300, 300)
FEATURE_MAP_SIZES = (19, 10, 5, 3, 2, 1)
COLORS = np.random.default_rng(42).integers(60, 256, (len(CLASSES), 3), dtype=np.uint8)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Mesmo preprocessamento da versao de imagens: RGB, NCHW, FP32 [-1,1]."""
    resized = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


def extract_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Identifica pontuacoes (3000,21) e offsets (3000,4)."""
    scores = offsets = None
    for output in outputs:
        array = np.asarray(output).squeeze().astype(np.float32)
        if array.ndim == 2 and array.shape[1] == len(CLASSES):
            scores = array
        elif array.ndim == 2 and array.shape[1] == 4:
            offsets = array

    if scores is None or offsets is None or len(scores) != len(offsets):
        raise RuntimeError(f"Saidas ONNX inesperadas: {[np.shape(x) for x in outputs]}")
    if not (np.isfinite(scores).all() and np.isfinite(offsets).all()):
        raise RuntimeError("A rede retornou valores NaN ou infinitos")

    sums = scores.sum(axis=1)
    if scores.min() < -1e-3 or scores.max() > 1.001 or not np.allclose(sums, 1.0, atol=0.02):
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        scores = exp / exp.sum(axis=1, keepdims=True)
    return scores, offsets


def create_anchors() -> np.ndarray:
    """Mesmas 3000 ancoras SSD validadas na versao de imagens."""
    min_scale, max_scale = 0.20, 0.95
    levels = len(FEATURE_MAP_SIZES)
    scales = [min_scale + (max_scale - min_scale) * i / (levels - 1)
              for i in range(levels)] + [1.0]
    all_anchors = []

    for level, side in enumerate(FEATURE_MAP_SIZES):
        s = scales[level]
        wh_pairs = [(s, s), (math.sqrt(s * scales[level + 1]),) * 2]
        for ratio in (2.0, 3.0):
            r = math.sqrt(ratio)
            wh_pairs.extend(((s * r, s / r), (s / r, s * r)))
        widths_heights = np.clip(np.asarray(wh_pairs, dtype=np.float32), 0, 1)
        centers_1d = (np.arange(side, dtype=np.float32) + 0.5) / side
        yy, xx = np.meshgrid(centers_1d, centers_1d, indexing="ij")
        centers = np.stack((xx, yy), axis=-1).reshape(-1, 2)
        anchors = np.concatenate((
            np.repeat(centers, 6, axis=0),
            np.tile(widths_heights, (side * side, 1)),
        ), axis=1)
        all_anchors.append(anchors)

    result = np.concatenate(all_anchors, axis=0)
    if result.shape != (3000, 4):
        raise AssertionError(f"Quantidade inesperada de ancoras: {result.shape}")
    return result


def decode_boxes(offsets: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Decodificacao SSD validada: offsets -> caixas xyxy normalizadas."""
    if offsets.shape != anchors.shape:
        raise ValueError(f"Offsets {offsets.shape} e ancoras {anchors.shape} incompativeis")
    center = anchors[:, :2] + offsets[:, :2] * anchors[:, 2:] / 10.0
    size = anchors[:, 2:] * np.exp(np.clip(offsets[:, 2:] / 5.0, -10.0, 10.0))
    return np.concatenate((center - size / 2.0, center + size / 2.0), axis=1)


def select_detections(
    scores: np.ndarray,
    boxes: np.ndarray,
    width: int,
    height: int,
    confidence: float,
    nms_threshold: float,
    max_detections: int = 100,
) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    """Filtra por classe, converte para pixels e aplica NMS por classe."""
    if boxes.shape != (len(scores), 4):
        raise ValueError("Quantidade de caixas incompativel com os scores")

    result = []
    scale = np.asarray((width, height, width, height), dtype=np.float32)
    for cls in range(1, len(CLASSES)):
        candidate_idx = np.flatnonzero(scores[:, cls] >= confidence)
        if len(candidate_idx) == 0:
            continue
        selected = np.clip(boxes[candidate_idx], 0.0, 1.0)
        valid = (np.isfinite(selected).all(axis=1)
                 & (selected[:, 2] > selected[:, 0])
                 & (selected[:, 3] > selected[:, 1]))
        candidate_idx, selected = candidate_idx[valid], selected[valid]
        if not len(candidate_idx):
            continue

        pixels = np.rint(selected * scale).astype(np.int32)
        rects, probs, coordinates = [], [], []
        for pos, (x1, y1, x2, y2) in enumerate(pixels):
            x1 = int(np.clip(x1, 0, width - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            rects.append([x1, y1, x2 - x1, y2 - y1])
            coordinates.append((x1, y1, x2, y2))
            probs.append(float(scores[candidate_idx[pos], cls]))

        if not rects:
            continue
        keep = np.asarray(cv2.dnn.NMSBoxes(rects, probs, confidence, nms_threshold)).reshape(-1)
        for index in keep:
            i = int(index)
            result.append((cls, probs[i], coordinates[i]))

    result.sort(key=lambda detection: detection[1], reverse=True)
    return result[:max_detections]


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Desenha caixas e classes sobre uma copia do quadro original."""
    output = image.copy()
    for cls, prob, (x1, y1, x2, y2) in detections:
        color = tuple(int(c) for c in COLORS[cls])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[cls]}: {prob:.1%}"
        cv2.putText(output, label, (x1, max(17, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output


def parse_source(value: str) -> int | str:
    """Permite --video 0/1 como webcam ou um caminho/URL como fonte."""
    return int(value) if value.isdecimal() else value


def create_writer(output: Path, capture: cv2.VideoCapture, width: int, height: int):
    """Abre arquivo de saida com mesmas dimensoes do video de entrada."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 1 or fps > 240:
        fps = 25.0  # webcams e streams podem nao informar FPS
    suffix = output.suffix.lower()
    if suffix in (".mp4", ".m4v", ".mov"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        raise ValueError("Formato de --output nao suportado: use .mp4, .mov ou .avi")
    writer = cv2.VideoWriter(str(output), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Nao foi possivel criar o video de saida: {output}")
    print(f"[INFO] Salvando video em {output} ({width}x{height}, {fps:.2f} FPS)")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deteccao de objetos em video com SSD MobileNetV1 ONNX e OpenCV DNN"
    )
    parser.add_argument("-v", "--video", default="0",
                        help="Webcam 0/1, arquivo de video ou URL RTSP/HTTP (padrao: 0)")
    parser.add_argument("-m", "--model", type=Path, required=True,
                        help="Modelo SSD MobileNetV1 PASCAL VOC da ST em ONNX")
    parser.add_argument("-c", "--confidence", type=float, default=0.3,
                        help="Confianca minima da deteccao (padrao: 0.3)")
    parser.add_argument("--nms", type=float, default=0.45,
                        help="Limiar IoU da supressao de caixas (padrao: 0.45)")
    parser.add_argument("--output", type=Path,
                        help="Salva video anotado (.mp4, .mov ou .avi)")
    parser.add_argument("--no-display", action="store_true",
                        help="Processa sem abrir janela (util para servidores)")
    parser.add_argument("--no-fps", action="store_true",
                        help="Nao desenha o FPS na imagem")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Para apos N quadros (0 = sem limite)")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Modelo ONNX nao encontrado: {args.model}")
    if not (0 < args.confidence < 1) or not (0 < args.nms < 1):
        parser.error("--confidence e --nms devem estar entre 0 e 1")
    if args.max_frames < 0:
        parser.error("--max-frames nao pode ser negativo")

    print(f"[INFO] Carregando ONNX: {args.model}")
    net = cv2.dnn.readNetFromONNX(str(args.model))
    output_names = net.getUnconnectedOutLayersNames()
    anchors = create_anchors()  # calculadas uma unica vez, nao por quadro

    source = parse_source(args.video)
    print(f"[INFO] Abrindo fonte: {source}")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {source}")

    writer = None
    count = 0
    fps_smoothed = 0.0
    start_total = time.perf_counter()
    try:
        print("[INFO] Pressione q ou ESC para encerrar (Ctrl+C no terminal).")
        while True:
            grabbed, frame = capture.read()
            if not grabbed or frame is None:
                print("[INFO] Fim do video ou nao foi possivel ler o proximo quadro.")
                break

            start_frame = time.perf_counter()
            net.setInput(preprocess(frame))
            outputs = net.forward(output_names)
            scores, offsets = extract_outputs(outputs)
            boxes = decode_boxes(offsets, anchors)
            height, width = frame.shape[:2]
            detections = select_detections(
                scores, boxes, width, height, args.confidence, args.nms
            )
            annotated = draw_detections(frame, detections)

            elapsed = max(time.perf_counter() - start_frame, 1e-9)
            instant_fps = 1.0 / elapsed
            fps_smoothed = instant_fps if count == 0 else (0.85 * fps_smoothed + 0.15 * instant_fps)
            if not args.no_fps:
                cv2.putText(annotated, f"FPS: {fps_smoothed:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if args.output is not None:
                if writer is None:
                    writer = create_writer(args.output, capture, width, height)
                writer.write(annotated)

            count += 1
            if count % 100 == 0:
                print(f"[INFO] Quadros: {count}, FPS processamento: {fps_smoothed:.1f}")

            if not args.no_display:
                cv2.imshow("MobileNetSSD ONNX - Video", annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if args.max_frames and count >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        runtime = time.perf_counter() - start_total
        print(f"[INFO] Encerrado. {count} quadros em {runtime:.2f} s.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, cv2.error) as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        sys.exit(1)
