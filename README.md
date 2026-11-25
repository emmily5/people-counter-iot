# Detector de Presença e Contador de Pessoas (IoT)

Projeto de IoT que usa uma **webcam** e **visão computacional** para:

- detectar se há pessoas no ambiente (presença);
- contar quantas pessoas a câmera está vendo;
- contar **entradas e saídas** quando alguém cruza uma linha virtual (a porta);
- acender um **LED** quando há presença (no notebook, um indicador na tela);
- registrar cada evento em um arquivo **CSV**, com data e hora.

A detecção usa o modelo **MobileNet-SSD** (formato ONNX), visto na Prática 9 da disciplina. O projeto parte do passo 2 da prática (detecção em vídeo em tempo real) e acrescenta o filtro de pessoas, o rastreamento, a contagem, a presença, o atuador e o registro.

A proposta completa (problema, solução, equipamentos, software e roteiro da apresentação) está em [PROPOSTA_PROJETO.md](PROPOSTA_PROJETO.md).

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `people_counter.py` | Programa principal: presença, contagem, rastreamento, LED e CSV. |
| `real_time_object_detection.py` | Detecção de objetos em vídeo (passo 2 da Prática 9). Suas funções são usadas pelo `people_counter.py`. |
| `models/ssd_mobilenetv1_pt_voc_300.onnx` | Rede neural MobileNet-SSD treinada no PASCAL VOC (entrada 300×300). |
| `eventos_contagem.csv` | Registro de eventos gerado pelo programa. |
| `PROPOSTA_PROJETO.md` | Proposta do projeto e roteiro. |

## Instalação

É preciso ter Python 3. O projeto foi testado com Python 3.14 e OpenCV 5.1.

```bash
git clone https://github.com/emmily5/people-counter-iot.git
cd people-counter-iot
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install numpy opencv-python
```

No Raspberry Pi, para usar o LED, instale também o `gpiozero`:

```bash
pip install gpiozero
```

## Como usar

```bash
python people_counter.py                                    # webcam padrão, linha vertical no meio
python people_counter.py --orientation horizontal --line 0.6
python people_counter.py -v video.mov --output contagem.mp4 # usa um arquivo de vídeo e grava o resultado
python people_counter.py --led-pin 17                       # Raspberry Pi com LED no GPIO17
```

Na janela do programa:

- `q` ou `ESC`: encerra;
- `r`: zera os contadores.

No macOS, se a imagem não aparecer, libere a câmera para o Terminal ou para o VS Code em *Ajustes do Sistema → Privacidade e Segurança → Câmera*.

### Opções

| Opção | Padrão | Descrição |
|---|---|---|
| `-v`, `--video` | `0` | Webcam (`0`, `1`...), arquivo de vídeo ou URL. |
| `-m`, `--model` | `models/ssd_mobilenetv1_pt_voc_300.onnx` | Caminho do modelo ONNX. |
| `-c`, `--confidence` | `0.5` | Confiança mínima para considerar uma detecção como pessoa. |
| `--nms` | `0.45` | Limite do NMS (remoção de caixas duplicadas). |
| `--orientation` | `vertical` | `vertical`: pessoas passam da esquerda para a direita; `horizontal`: de cima para baixo. |
| `--line` | `0.5` | Posição da linha virtual, como fração da largura ou da altura. |
| `--margin` | `0.08` | Meia-largura da faixa de tolerância em volta da linha. Evita contagens repetidas de quem para sobre ela. |
| `--absence` | `3.0` | Segundos sem pessoas até o estado voltar para "LIVRE". |
| `--log` | `eventos_contagem.csv` | Arquivo CSV de eventos. |
| `--led-pin` | — | Pino GPIO do LED de presença (só no Raspberry Pi). |
| `--output` | — | Grava o vídeo anotado (`.mp4`, `.mov` ou `.avi`). |
| `--no-display` | — | Roda sem abrir janela. |

## Como funciona

Para cada quadro da câmera, o programa:

1. **Pré-processa** a imagem: redimensiona para 300×300, converte para RGB e normaliza para [-1, 1].
2. **Executa a rede** MobileNet-SSD com o módulo `cv2.dnn` do OpenCV.
3. **Filtra** as detecções: mantém só a classe `person` com confiança acima do limite e remove caixas duplicadas (NMS).
4. **Rastreia** cada pessoa por centroide, mantendo o mesmo ID entre quadros.
5. **Conta** uma ENTRADA ou uma SAÍDA quando um ID passa de um lado para o outro da linha virtual.
6. **Atualiza a presença**: começa quando alguém aparece e termina depois de N segundos sem ninguém. O LED acompanha esse estado.
7. **Registra e exibe**: grava os eventos no CSV e mostra na tela as caixas, os IDs, a linha, os contadores e o FPS.

### Formato do CSV

```
data_hora,evento,pessoas_no_quadro,entradas,saidas,ocupacao_estimada
2026-11-23T17:11:22,PRESENCA_INICIO,1,0,0,0
```

Eventos possíveis: `PRESENCA_INICIO`, `PRESENCA_FIM`, `ENTRADA`, `SAIDA` e `RESET`. A ocupação estimada é `entradas − saídas`.

## Hardware

| Equipamento | Função |
|---|---|
| Raspberry Pi 4/5 (no protótipo, um notebook) | Processamento: vídeo, rede neural, contagem e registro. |
| Webcam USB ou câmera do notebook | Sensor: capta as imagens do ambiente. |
| LED + resistor de 220 Ω no GPIO17 | Atuador: acende quando há presença. |

Ligação do LED: GPIO17 → resistor de 220 Ω → perna maior do LED; perna menor do LED → GND.

## Limitações

- A contagem pode errar com muitas pessoas sobrepostas, pouca luz ou a câmera mal posicionada.
- O protótipo foi testado só em notebook. No Raspberry Pi o FPS tende a ser menor.
- Relé e buzzer (alerta de lotação) estão previstos na proposta, mas ainda não foram implementados.

## Privacidade

Todo o processamento é feito no próprio dispositivo. Por padrão, nenhuma imagem é enviada ou guardada: o CSV contém apenas números e horários. O vídeo só é gravado se a opção `--output` for usada.
