# Detector de Presença e Contador de Pessoas com Visão Computacional

## 1. Problema

Muitos ambientes precisam saber **se há pessoas** e **quantas pessoas** estão no local: salas de aula, laboratórios, bibliotecas, lojas e recepções. Hoje essa informação costuma vir de três fontes, e todas têm limitações:

- **Contagem manual:** cansativa, sujeita a erros e sem registro histórico.
- **Sensores PIR (infravermelho passivo):** só detectam movimento. Uma pessoa parada (estudando, por exemplo) "desaparece", e o sensor não diz quantas pessoas existem.
- **Catracas e sensores de feixe:** caros, exigem obra e podem contar errado quando duas pessoas passam juntas.

Sem esses dados não é possível:
- controlar a **lotação máxima** de um ambiente;
- desligar **luz e ar-condicionado** em salas vazias, o que desperdiça energia;
- conhecer os **horários de pico**, para planejar limpeza, atendimento e segurança.

## 2. Solução proposta

Um dispositivo IoT de baixo custo que usa **uma webcam** e **visão computacional com redes neurais** para:

1. **Detectar presença:** indica se há ou não pessoas no ambiente. O estado só muda para "livre" depois de alguns segundos sem ninguém, o que evita oscilações.
2. **Contar pessoas no quadro:** mostra em tempo real quantas pessoas a câmera está vendo.
3. **Contar entradas e saídas:** cada pessoa recebe um ID e é acompanhada pelo **rastreamento de centroides**. Quando cruza uma **linha virtual** (a porta), o sistema conta uma entrada ou uma saída. A ocupação estimada é `entradas − saídas`.
4. **Acionar um atuador:** um LED acende quando há presença. No protótipo em notebook, o LED é simulado por um indicador vermelho/verde na tela; no Raspberry Pi, o código aciona um LED real no GPIO. Em uma versão futura, a mesma saída poderia ligar um relé de iluminação ou disparar um alerta de lotação.
5. **Registrar dados:** cada evento (início/fim de presença, entrada, saída) é gravado em um arquivo **CSV** com data e hora. Esse arquivo pode alimentar planilhas ou dashboards.

A detecção usa o modelo **MobileNet-SSD** (formato ONNX) visto na **Prática 9**. Esse modelo foi criado para rodar em dispositivos móveis e embarcados e reconhece 20 classes do conjunto PASCAL VOC. O projeto usa só a classe **`person`**. O passo 2 da prática (adaptar o código para vídeo em tempo real) é o ponto de partida: o projeto acrescenta o filtro de pessoas, o rastreamento, a contagem, a presença, o atuador e o registro.

**Vantagens:**
- detecta pessoas paradas, ao contrário do PIR;
- conta várias pessoas ao mesmo tempo;
- hardware barato;
- **privacidade:** o processamento é local (edge computing). Em operação normal nenhuma imagem é enviada ou guardada, só números; a gravação de vídeo existe apenas como opção de teste (`--output`) e fica desligada por padrão.

## 3. Equipamentos

| Equipamento | Função |
|---|---|
| **SBC Raspberry Pi 4/5** (no protótipo atual, um notebook) | Unidade de processamento: captura o vídeo, executa a rede neural (OpenCV DNN), faz o rastreamento e a contagem, controla o atuador e grava os registros. |
| **Webcam USB** (720p, ou a câmera do notebook) | **Sensor** do sistema: capta as imagens do ambiente. Deve ficar no alto, apontada para a porta ou passagem. |
| **LED + resistor de 220 Ω** (GPIO17) | **Atuador:** acende quando há presença. No notebook, é simulado por um indicador vermelho/verde na tela. |
| **Módulo relé 5 V** (opcional, extensão futura) | Atuador de potência: ligaria/desligaria iluminação ou ar-condicionado conforme a presença. Ainda não implementado no código. |
| **Buzzer** (opcional, extensão futura) | Alerta sonoro quando a ocupação passasse do limite. Ainda não implementado no código. |
| **Cartão microSD (32 GB) + fonte 5 V/3 A** | Sistema operacional, modelo, código e registros CSV; alimentação da placa. |
| **Protoboard e jumpers** | Montagem do circuito do LED (e, futuramente, do relé e do buzzer). |

## 4. Software e código

**Software utilizado:**
- **Python 3** como linguagem principal.
- **OpenCV (módulo `cv2.dnn`)** para capturar o vídeo da webcam, executar a rede neural ONNX e desenhar na tela.
- **NumPy** para os cálculos com matrizes: pré-processamento, decodificação das caixas e distâncias.
- **Modelo `ssd_mobilenetv1_pt_voc_300.onnx`**: MobileNet-SSD treinado no PASCAL VOC, entrada de 300×300 pixels.
- **gpiozero** (só no Raspberry Pi) para controlar o LED.
- **CSV** para guardar o histórico de eventos.

**Código desenvolvido** (`people_counter.py`): reaproveita as funções da Prática 9 (`preprocess`, `extract_outputs`, `create_anchors`, `decode_boxes`, `select_detections`). O funcionamento, quadro a quadro:

1. **Captura** o quadro da webcam (`cv2.VideoCapture`).
2. **Pré-processa** o quadro: redimensiona para 300×300, converte para RGB e normaliza os valores para [-1, 1].
3. **Faz a inferência** com o MobileNet-SSD. A rede devolve 3000 caixas candidatas, cada uma com uma pontuação para cada uma das 21 classes.
4. **Decodifica as caixas** e **filtra**: mantém só a classe `person` com confiança ≥ 0,5 e remove caixas duplicadas (NMS).
5. **Rastreia:** a classe `CentroidTracker` liga cada pessoa detectada ao centroide mais próximo do quadro anterior, mantendo o mesmo ID.
6. **Conta:** quando um ID troca de lado da linha virtual, o sistema registra uma ENTRADA ou uma SAÍDA.
7. **Detecta presença:** a presença começa assim que alguém aparece e termina após N segundos sem pessoas. Nesses momentos o LED liga ou desliga.
8. **Registra e exibe:** grava os eventos no CSV e mostra na tela as caixas, os IDs, a linha, os contadores, o estado e o FPS.

**Como executar:**
```bash
pip install numpy opencv-python
python people_counter.py                           # webcam padrão
python people_counter.py --orientation horizontal  # linha horizontal (pessoas passando de cima p/ baixo)
python people_counter.py --led-pin 17              # Raspberry Pi com LED
```
Teclas: `q`/`ESC` encerra; `r` zera os contadores.

**Limitações conhecidas:** a contagem pode errar quando há muitas pessoas sobrepostas, pouca luz ou câmera mal posicionada. O protótipo foi testado em notebook; no Raspberry Pi o FPS tende a ser menor, o que pode exigir ajustes (menor resolução, confiança, tempo de ausência). Melhorias futuras: acionar o relé e o buzzer de lotação, enviar os dados por MQTT para uma plataforma IoT (dashboard na nuvem) e usar um modelo mais recente (por exemplo, YOLO).

---

# Roteiro

## A) Roteiro de desenvolvimento

1. **Ambiente:** instalar Python, OpenCV e NumPy; rodar `dl_object_detection.py` da Prática 9 nas imagens de exemplo.
2. **Tempo real:** rodar `real_time_object_detection.py` com a webcam (passo 2 da prática) e anotar o FPS.
3. **Filtro de pessoas:** manter só a classe `person` e mostrar a quantidade de pessoas no quadro.
4. **Presença:** criar o estado PRESENÇA/LIVRE com tempo de tolerância e um indicador na tela.
5. **Rastreamento e linha virtual:** criar os IDs por centroide e contar entradas e saídas.
6. **Registro:** gravar os eventos em CSV.
7. **Hardware:** passar para o Raspberry Pi e ligar o LED ao GPIO17 (com o resistor de 220 Ω até o GND).
8. **Testes:** testar com 1, 2 e 3 pessoas, de frente e de costas, com luz forte e fraca. Comparar com uma contagem manual e calcular a taxa de acerto.
9. **Ajustes:** calibrar a confiança (`-c`), a posição da linha (`--line`) e o tempo de ausência (`--absence`).

## B) Roteiro da apresentação (~5 min)

| Tempo | Conteúdo |
|---|---|
| 0:00–0:45 | **Problema:** controle de lotação e desperdício de energia; por que o sensor PIR não basta (pessoa parada, não conta). |
| 0:45–1:30 | **Solução:** webcam + IA que detecta presença, conta pessoas e registra entradas/saídas; processamento local, com privacidade. |
| 1:30–2:15 | **Equipamentos:** Raspberry Pi (processamento), webcam (sensor), LED/relé (atuador), com a função de cada um. |
| 2:15–3:15 | **Software:** MobileNet-SSD em ONNX da Prática 9 → filtro `person` → rastreamento por centroide → linha virtual → CSV. Mostrar o fluxograma ou um trecho do código. |
| 3:15–4:30 | **Demonstração ao vivo:** 1) sala vazia: indicador verde "LIVRE"; 2) uma pessoa entra: "PRESENÇA", LED aceso, Entradas = 1; 3) segunda pessoa: "No quadro: 2"; 4) uma sai: Saídas = 1; 5) abrir o CSV gerado. |
| 4:30–5:00 | **Resultados e conclusão:** taxa de acerto nos testes, FPS obtido, limitações e melhorias futuras (MQTT, dashboard, YOLO). |
