# Pipeline do Sonar de Imagem (OceanSimImagingSonar)

Este documento descreve o funcionamento completo do sonar de imagem acústica no PatoSim,
cobrindo a cadeia desde o backend de física do OceanSim até a gravação e o preview na UI.

---

## 1. Sensor físico modelado

O sensor é baseado no **Oculus M370s / MT370s** — sonar de imagem acústica de feixe largo
(fan-beam). Parâmetros de fábrica modelados:

| Parâmetro | Valor padrão | Descrição |
|---|---|---|
| `min_range` | 0.2 m | Distância mínima de detecção |
| `max_range` | 10.0 m (3.0 m no OceanSim) | Alcance máximo |
| `range_res` | 0.005–0.008 m | Resolução em distância |
| `hori_fov` | 130° | Campo de visão horizontal |
| `vert_fov` | 20° | Campo de visão vertical |
| `angular_res` | 0.25–0.5° | Resolução angular |
| `hori_res` | 3000–4000 px | Resolução da câmera raytrace |

A resolução vertical é derivada automaticamente pela razão `hori_fov / vert_fov` para
manter pixels quadrados (limitação do Isaac Sim).

---

## 2. Arquitetura do pipeline

```
Simulação físics (PhysX)
        │
        ▼
ImagingSonarSensor (isaacsim/oceansim/sensors/ImagingSonarSensor.py)
  │  Herda de Camera (Isaac Sim)
  │  Cria render product com resolução (hori_res × vert_res)
  │
  ├─ Annotators (Isaac Replicator):
  │     pointcloud         → (N,3) XYZ world + normais + semantics (Warp GPU)
  │     CameraParams       → matriz de view transform 4×4
  │     semantic_segmentation → idToLabels (reflectividade por material)
  │
  ▼ make_sonar_data()
  GPU (Warp kernels — ImagingSonar_kernels.py)
  │
  ├─ compute_intensity: incidência angular + reflectividade semântica + atenuação
  ├─ world2local: transforma PCL para coordenadas do sonar (esférico)
  ├─ bin_intensity: acumula intensidade em grid (r × azimute)
  ├─ Normalização: por máximo global ("all") ou por máximo por faixa de range ("range")
  ├─ Ruído gaussiano multiplicativo (gau_noise_param)
  ├─ Ruído de Rayleigh range-dependente + streak central (ray_noise_param, central_peak)
  └─ make_sonar_map_*: converte (r, azi) → ponto Cartesiano (x,y) + intensidade z
        resultado: sonar_map (N_range × N_azi, vec3)
        onde vec3 = (x_cart, y_cart, intensity)
  │
  ▼ make_sonar_image()
  Warp kernel: sonar_map[:,:,2] → imagem grayscale RGBA (N_range × N_azi × 4)
        Obs: o eixo de azimute é invertido (width-j) para correção de orientação
  │
  ▼ OceanSimImagingSonar.update_state()  [sensors.py]
        rgb_image.set_value(sonar_np[...,:3])  → Buffer com tag "rgb"
```

### Observação crítica sobre o formato de saída

O `make_sonar_image()` retorna uma **imagem retangular** onde:
- **Linhas** = bins de range (distância crescente de cima para baixo)
- **Colunas** = bins de azimute (ângulo crescente da direita para a esquerda)

Esta representação é um grid polar (r × azi) **não** uma imagem Cartesiana.
Exibir diretamente como imagem retangular produz distorção visual porque o
espaço polar não é linear em X,Y. Objetos próximos ao centro aparecem
"comprimidos" e objetos laterais aparecem "esticados".

---

## 3. Integração com o PatoSim (sensors.py)

**Arquivo:** `exts/omni.ext.patosim/omni/ext/patosim/sensors.py`

### Classe `OceanSimImagingSonar`

```python
class OceanSimImagingSonar(Sensor):
    rgb_image = Buffer(tags=["rgb"])      # imagem polar r×azi RGBA
    pointcloud = Buffer(tags=["pointcloud"])  # PCL bruto do annotator
    position = Buffer()
    orientation = Buffer()
    status = Buffer("idle")
```

**Método `build()`**: instancia `ImagingSonarSensor` com os parâmetros do robô
(veja `OceanSimROVRobot.sonar_max_range` em `robots.py`).

**Método `_ensure_initialized()`**: chama `sonar_initialize(viewport=False,
include_unlabelled=True)` — `include_unlabelled=True` permite que o sonar
"veja" meshes sem labels semânticos (essencial para estruturas da cena sem
reflectividade configurada).

**Método `update_state()`** (chamado a cada physics tick):
1. `make_sonar_data()` — calcula mapa acústico na GPU
2. `make_sonar_image()` → `rgb_image.set_value(...)` — grava no buffer
3. `scan_data["pcl"]` → `pointcloud.set_value(...)` — PCL bruto (opcional)

**Métodos de preview** (adicionados ao PatoSim):
- `render_polar_preview(size)`: converte grid r×azi → imagem Cartesiana bird's-eye (fan shape)
- `render_planar_preview(width, height)`: redimensiona o grid bruto para visualização direta

---

## 4. Integração com o sistema de gravação (MobilityGen/PatoSim)

### Fase online (gravação em tempo real)

O sonar **não** é gravado durante a fase online de navegação. O buffer `rgb_image`
tem a tag `"rgb"`, mas a função `state_dict_rgb()` não é chamada online porque
`deferred_sensor_processing_enabled = True` (valor padrão).

O que é gravado online:
- `state/common/*.npy` — pose do sonar (`position`, `orientation` via `state_dict_common`)
- Status do sonar não é gravado (buffer `status` não tem tag relevante para escrita)

### Fase offline (replay)

O sonar é capturado **como parte do replay de câmeras**, não como módulo independente:

1. `replay_implementation.py` descobre módulos de câmera via `discover_camera_modules()`
2. `_is_camera_module()` **NÃO detecta** o sonar (sem `raw_rgb_image`, `disable_rendering`)
3. O sonar é atualizado implicitamente: `_advance_replay_step()` → `scenario.write_replay_data()` → `robot.update_state()` → `sonar.update_state()`
4. `writer.write_state_dict_rgb(scenario.state_dict_rgb(), step)` captura todos os buffers com tag `"rgb"`, **incluindo** `sonar.rgb_image`
5. Imagens salvas em: `state/rgb/sonar/XXXXXXXX.jpg`

**Implicação**: o sonar depende do `rgb_enabled` para produzir imagem no replay.
Ele é ativado quando `scenario.enable_rgb_rendering()` é chamado no início do replay
(via recursão em `common.py`).

### Diferença em relação ao LiDAR

| | Sonar | LiDAR |
|---|---|---|
| Captura online | Não (deferred) | Não (deferred) |
| Replay: modulo separado | Não | Sim (via `drive_rtx_lidar_render_products`) |
| Gravado via | `state_dict_rgb()` | `state_dict_pointcloud()` |
| Formato de saída | JPEG (rgb) | PLY / NPY (pointcloud) |
| Aquecimento (warm-up) | Não necessário | Sim (2–3 frames RTX) |

---

## 5. Parâmetros configuráveis em `OceanSimROVRobot` (robots.py)

```python
sonar_translation = (0.3, 0.0, 0.3)       # posição relativa ao corpo do ROV (m)
sonar_orientation_euler_deg = (0.0, 0.0, 0.0)  # orientação (graus)
sonar_max_range = 10.0                     # alcance máximo (m)
enable_sonar = True                        # habilitar/desabilitar sensor
```

Os demais parâmetros físicos (`hori_fov`, `angular_res`, `range_res`) usam os
defaults do `OceanSimImagingSonar.build()` que espelham o hardware Oculus M370s.

---

## 6. Preview na UI

### Janela Sensor Preview (geral)

- Exibe a imagem bruta do sonar (grid r×azi) ao lado das câmeras
- Seção "Sonar" no painel lateral da janela `PatoSim - Sensor Preview`
- Atualizada a cada N frames (controlado por `_preview_update_interval_frames`)
- Ativada pelo toggle "Preview de Sensores" no Window Manager

### Janela Sonar Preview (dedicada) — adicionada ao PatoSim

- Acessível via "Preview Sonar (polar/planar)" no Window Manager
- Janela flutuante `PatoSim - Sonar Preview` (440×480 px)
- **Modo polar**: converte grid r×azi → imagem Cartesiana (bird's-eye, formato fan)
  - X (vertical) = distância à frente do sonar
  - Y (horizontal) = posição lateral
  - Estruturas aparecem na posição física correta
- **Modo planar**: exibe o grid r×azi bruto (útil para debug)
- Tamanho da imagem configurável na janela

---

## 7. Diagnóstico e problemas comuns

### Sonar não produz imagem

1. **`enable_sonar = False`** em `OceanSimROVRobot` → sonar não é instanciado
2. **`enable_rgb_rendering()` não chamado** → `_rgb_enabled = False`, `update_state()` não gera imagem
3. **Cena sem geometria no FOV** → `semanticSeg_annot.get_data()` vazio → `scan()` retorna `False`
4. **Reflectividade não aplicada** → intensidade nula → imagem toda preta

### Imagem distorcida (problema original relatado)

- Causa: a imagem do sonar é um grid polar exibido como retângulo
- Solução: usar o modo **polar** no `PatoSim - Sonar Preview` para ver a projeção Cartesiana correta

### Sonar não salvo no replay

- Verificar que `rgb_enabled` está ativo no script de replay (`--rgb`)
- O sonar é capturado junto com as câmeras em `state/rgb/sonar/`

---

## 8. Melhorias propostas

### Curto prazo

1. **Colormap acústico**: mapear intensidade para escala de cor (tipo sonar real — branco/azul/vermelho) em vez de grayscale
2. **Overlay de escala**: adicionar marcações de distância (arcos concêntricos) e ângulo no preview polar
3. **Parâmetros de ruído na UI**: expor `gau_noise_param`, `ray_noise_param`, `attenuation` na janela de preview para ajuste em tempo real

### Médio prazo

4. **Gravação polar no replay**: salvar a imagem Cartesiana (polar view) como segundo canal alongside o grid bruto — elimina a necessidade de pós-processamento offline
5. **Sonar como módulo de câmera no replay**: modificar `_is_camera_module()` para reconhecer `OceanSimImagingSonar` → permite pass de sonar separado, controlado por `--sonar` flag
6. **Buffer de sonar data**: expor `sonar_map` (grid float32 de intensidades) como buffer `npy` separado além do JPEG — permite análise quantitativa posterior

### Longo prazo

7. **Multi-beam 3D**: a API do OceanSim suporta captura de nuvem de pontos do sonar (`scan_data["pcl"]`). Combinar PCL do sonar com pose 6-DOF do ROV gera mapa 3D acústico — útil para SLAM subaquático
8. **Frequência assíncrona**: o sonar real opera a 40 Hz max. O PatoSim processa o sonar em todo physics tick (100 Hz) — implementar decimação via `sonar_update_interval` para respeitar o modelo físico

---

## 9. Referências de código

| Arquivo | Conteúdo |
|---|---|
| `isaacsim/oceansim/sensors/ImagingSonarSensor.py` | Backend do sensor (Camera + Warp) |
| `isaacsim/oceansim/utils/ImagingSonar_kernels.py` | Kernels GPU: binning, ruído, conversão |
| `exts/omni.ext.patosim/omni/ext/patosim/sensors.py` | Wrapper PatoSim: `OceanSimImagingSonar` |
| `exts/omni.ext.patosim/omni/ext/patosim/robots.py` | Configuração e instanciação do sonar no ROV |
| `exts/omni.ext.patosim/omni/ext/patosim/extension.py` | Preview UI, janela dedicada, toggle |
| `scripts/replay_implementation.py` | Captura offline via `state_dict_rgb()` |
| `exts/omni.ext.patosim/omni/ext/patosim/writer.py` | Persistência: `write_state_dict_rgb()` → JPEG |
