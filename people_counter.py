"""Detector de Presenca e Contador de Pessoas com webcam (MobileNetSSD ONNX).

Reaproveita o pipeline da Pratica 9 (real_time_object_detection.py) e acrescenta:
  * filtro da classe "person";
  * contagem de pessoas visiveis no quadro (ocupacao instantanea);
  * deteccao de presenca com tempo de tolerancia (evita "piscar");
  * rastreamento por centroide + linha virtual para contar ENTRADAS e SAIDAS;
  * registro de eventos em CSV (data/hora, evento, contadores);
  * atuador: indicador na tela e, opcionalmente, LED via GPIO (Raspberry Pi).

Requisitos: pip install numpy opencv-python   (opcional no Raspberry: gpiozero)

Exemplos:
    python people_counter.py                          # webcam 0, linha vertical no meio
    python people_counter.py -v 1 --orientation horizontal --line 0.6
    python people_counter.py -v imd1039.mov --output contagem.mp4
    python people_counter.py --led-pin 17             # acende LED no GPIO17 se houver presenca

Pressione q ou ESC para sair; r para zerar os contadores.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from real_time_object_detection import (
    CLASSES,
    create_anchors,
    create_writer,
    decode_boxes,
    extract_outputs,
    parse_source,
    preprocess,
    select_detections,
)

PERSON = CLASSES.index("person")
GREEN, RED, YELLOW, WHITE, BLACK = (0, 200, 0), (0, 0, 230), (0, 220, 255), (255, 255, 255), (0, 0, 0)


class CentroidTracker:
    """Rastreador simples: associa cada deteccao ao centroide mais proximo."""

    def __init__(self, max_disappeared: int = 15, max_distance: float = 120.0):
        self.next_id = 0
        self.objects: dict[int, tuple[int, int]] = {}
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, centroids: list[tuple[int, int]]) -> dict[int, tuple[int, int]]:
        ids = list(self.objects)
        # Todos os pares (distancia, id, indice), do mais proximo ao mais distante.
        pairs = sorted(
            (math.dist(self.objects[oid], c), oid, j)
            for oid in ids for j, c in enumerate(centroids)
        )
        used_ids, used_cols = set(), set()
        for dist, oid, j in pairs:
            if dist > self.max_distance or oid in used_ids or j in used_cols:
                continue
            self.objects[oid] = centroids[j]
            self.disappeared[oid] = 0
            used_ids.add(oid)
            used_cols.add(j)

        for oid in ids:
            if oid not in used_ids:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
        for j, c in enumerate(centroids):
            if j not in used_cols:
                self._register(c)
        return self.objects


class EventLog:
    """Grava eventos em CSV para analise posterior (planilha, dashboard...)."""

    def __init__(self, path: Path | None):
        self.file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            self.file = path.open("a", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            if new:
                self.writer.writerow(["data_hora", "evento", "pessoas_no_quadro",
                                      "entradas", "saidas", "ocupacao_estimada"])

    def write(self, event: str, visible: int, entries: int, exits: int):
        stamp = datetime.now().isoformat(timespec="seconds")
        print(f"[EVENTO] {stamp} {event}: quadro={visible} entradas={entries} saidas={exits}")
        if self.file:
            self.writer.writerow([stamp, event, visible, entries, exits, max(entries - exits, 0)])
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


class PresenceActuator:
    """LED fisico (gpiozero) quando disponivel; caso contrario, apenas simulado."""

    def __init__(self, pin: int | None):
        self.led = None
        if pin is not None:
            try:
                from gpiozero import LED  # so existe no Raspberry Pi
                self.led = LED(pin)
                print(f"[INFO] LED de presenca no GPIO{pin}")
            except Exception as exc:  # noqa: BLE001
                print(f"[AVISO] GPIO indisponivel ({exc}); usando indicador apenas na tela.")

    def set(self, on: bool):
        if self.led is not None:
            self.led.on() if on else self.led.off()

    def close(self):
        if self.led is not None:
            self.led.off()
            self.led.close()


def draw_overlay(frame, people, tracked, line, margin, horizontal, present, entries, exits, fps):
    h, w = frame.shape[:2]
    # Faixa de tolerancia (translucida) em volta da linha virtual.
    band = frame.copy()
    if horizontal:
        cv2.rectangle(band, (0, line - margin), (w, line + margin), (255, 0, 255), -1)
    else:
        cv2.rectangle(band, (line - margin, 0), (line + margin, h), (255, 0, 255), -1)
    cv2.addWeighted(band, 0.15, frame, 0.85, 0, dst=frame)
    for _, prob, (x1, y1, x2, y2) in people:
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
        cv2.putText(frame, f"pessoa {prob:.0%}", (x1, max(17, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)
    for oid, (cx, cy) in tracked.items():
        cv2.circle(frame, (cx, cy), 5, YELLOW, -1)
        cv2.putText(frame, f"ID {oid}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2, cv2.LINE_AA)

    p1, p2 = ((0, line), (w, line)) if horizontal else ((line, 0), (line, h))
    cv2.line(frame, p1, p2, (255, 0, 255), 2)

    cv2.rectangle(frame, (0, 0), (260, 130), BLACK, -1)
    status, color = ("PRESENCA", RED) if present else ("LIVRE", GREEN)
    cv2.circle(frame, (20, 22), 10, color, -1)  # "LED" virtual
    lines = [(status, color), (f"No quadro: {len(people)}", WHITE),
             (f"Entradas: {entries}  Saidas: {exits}", WHITE),
             (f"Ocupacao estimada: {max(entries - exits, 0)}", WHITE),
             (f"FPS: {fps:.1f}", WHITE)]
    for i, (text, c) in enumerate(lines):
        cv2.putText(frame, text, (40 if i == 0 else 10, 28 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector de presenca e contador de pessoas")
    parser.add_argument("-v", "--video", default="0", help="Webcam 0/1, arquivo ou URL (padrao: 0)")
    parser.add_argument("-m", "--model", type=Path,
                        default=Path(__file__).with_name("models") / "ssd_mobilenetv1_pt_voc_300.onnx")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="Confianca minima para 'person' (padrao: 0.5)")
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--orientation", choices=("vertical", "horizontal"), default="vertical",
                        help="vertical: pessoas cruzam da esquerda p/ direita; horizontal: de cima p/ baixo")
    parser.add_argument("--line", type=float, default=0.5,
                        help="Posicao da linha virtual (fracao da largura/altura, padrao 0.5)")
    parser.add_argument("--margin", type=float, default=0.08,
                        help="Meia-largura da faixa de tolerancia em volta da linha (fracao, padrao 0.08)")
    parser.add_argument("--absence", type=float, default=3.0,
                        help="Segundos sem pessoas para encerrar a presenca (padrao 3)")
    parser.add_argument("--log", type=Path, default=Path(__file__).with_name("eventos_contagem.csv"),
                        help="Arquivo CSV de eventos")
    parser.add_argument("--led-pin", type=int, help="GPIO do LED de presenca (Raspberry Pi)")
    parser.add_argument("--output", type=Path, help="Salva video anotado (.mp4/.mov/.avi)")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Modelo ONNX nao encontrado: {args.model}")
    if not 0 < args.line < 1:
        parser.error("--line deve estar entre 0 e 1")

    net = cv2.dnn.readNetFromONNX(str(args.model))
    output_names = net.getUnconnectedOutLayersNames()
    anchors = create_anchors()

    capture = cv2.VideoCapture(parse_source(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {args.video}")

    horizontal = args.orientation == "horizontal"
    tracker = CentroidTracker()
    last_side: dict[int, int] = {}
    log = EventLog(args.log)
    actuator = PresenceActuator(args.led_pin)
    writer = None
    entries = exits = 0
    present = False
    last_seen = 0.0
    fps = 0.0

    try:
        print("[INFO] q/ESC para sair, r para zerar contadores.")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            t0 = time.perf_counter()
            h, w = frame.shape[:2]
            line = int((h if horizontal else w) * args.line)

            net.setInput(preprocess(frame))
            scores, offsets = extract_outputs(net.forward(output_names))
            detections = select_detections(scores, decode_boxes(offsets, anchors),
                                            w, h, args.confidence, args.nms)
            people = [d for d in detections if d[0] == PERSON]

            centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for _, _, (x1, y1, x2, y2) in people]
            tracked = tracker.update(centroids)

            # Cruzamento da linha: troca de lado = entrada (-> / para baixo) ou saida.
            # Faixa morta em volta da linha: quem fica parado sobre ela nao gera
            # contagens repetidas; so conta ao sair totalmente para o outro lado.
            margin = int((h if horizontal else w) * args.margin)
            for oid, (cx, cy) in tracked.items():
                pos = cy if horizontal else cx
                if abs(pos - line) <= margin:
                    continue  # dentro da faixa: mantem o ultimo lado conhecido
                side = 1 if pos > line else 0
                prev = last_side.get(oid)
                if prev is not None and side != prev:
                    if side == 1:
                        entries += 1
                        log.write("ENTRADA", len(people), entries, exits)
                    else:
                        exits += 1
                        log.write("SAIDA", len(people), entries, exits)
                last_side[oid] = side
            for oid in list(last_side):
                if oid not in tracked:
                    del last_side[oid]

            # Presenca com histerese temporal.
            now = time.monotonic()
            if people:
                last_seen = now
                if not present:
                    present = True
                    actuator.set(True)
                    log.write("PRESENCA_INICIO", len(people), entries, exits)
            elif present and now - last_seen > args.absence:
                present = False
                actuator.set(False)
                log.write("PRESENCA_FIM", 0, entries, exits)

            inst = 1.0 / max(time.perf_counter() - t0, 1e-9)
            fps = inst if fps == 0 else 0.85 * fps + 0.15 * inst
            annotated = draw_overlay(frame, people, tracked, line, margin, horizontal,
                                     present, entries, exits, fps)

            if args.output is not None:
                if writer is None:
                    writer = create_writer(args.output, capture, w, h)
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Contador de Pessoas", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    entries = exits = 0
                    log.write("RESET", len(people), entries, exits)
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        actuator.close()
        log.close()
        print(f"[INFO] Total: {entries} entradas, {exits} saidas.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, cv2.error) as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        sys.exit(1)
