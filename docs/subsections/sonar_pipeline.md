# Pipeline do Sonar de Imagem (OceanSimImagingSonar)

Este documento descreve o funcionamento completo do sonar de imagem acústica no PatoSim,
cobrindo a cadeia desde o backend de física do OceanSim até a gravação e os previews na UI.

---

## 1. Sensor físico modelado

O sensor é baseado no **Oculus M370s / MT370s** — sonar de imagem acústica de feixe largo
(fan-beam). Parâmetros de fábrica modelados:

| Parâmetro | Valor PatoSim | Descrição |
|---|---|---|
| `min_range` | 0.2 m | Distância mínima de detecção |
| `max_range` | 10.0 m | Alcance máximo (configurável via `sonar_max_range`) |
| `range_res` | 0.005 m | Resolução em distância → N_range = (10.0−0.2)/0.005 = **1960 bins** |
| `hori_fov` | 130° | Campo de visão horizontal |
| `vert_fov` | 20° | Campo de visão vertical |
| `angular_res` | 0.25° | Resolução angular → N_azi = 130/0.25 = **520 bins** |
| `hori_res` | 4000 px | Resolução da câmera raytrace (render product) |

A resolução vertical é derivada automaticamente pela razão `hori_fov / vert_fov` para
manter pixels quadrados (limitação do Isaac Sim).

**Proporção natural do grid bruto:** N_range / N_azi = 1960 / 520 ≈ **3.77 : 1** (altura : largura).

---

## 2. Arquitetura do pipeline

```
Simulação física (PhysX)
        │
        ▼
ImagingSonarSensor (isaacsim/oceansim/sensors/ImagingSonarSensor.py)
  │  Herda de Camera (Isaac Sim)
  │  Cria render product com resolução (hori_res × vert_res)
  │
  ├─ Annotators (Isaac Replicator):
  │     pointcloud         → (N,3) XYZ world + normais + semantics (Warp GPU)
  │     CameraParams       → viewTransform 4×4 (world→camera)
  │     semantic_segmentation → idToLabels (reflectividade por material)
  │
  ▼ make_sonar_data()
  GPU (Warp kernels — ImagingSonar_kernels.py)
  │
  ├─ world2local:     PCL world → frame sonar (rotação de eixos USD→sonar)
  │     pcl_local = (camera_X, -camera_Z, camera_Y)
  │     azi = atan2(forward, lateral_right)  → azi=90° = frente
  │
  ├─ compute_intensity: incidência angular × reflectividade semântica × exp(-atten×dist)
  ├─ bin_intensity:   acumula intensidade em grid (N_range × N_azi)
  │     i = range bin, j = azimuth bin (j=0 = azi_min = 25° = DIREITA)
  │
  ├─ binning_method "sum" | "mean": acumulação por bin
  ├─ normalizing_method "all" | "range": normalização por máximo global ou por faixa
  ├─ Ruído gaussiano multiplicativo (gau_noise_param) — kernel normal_2d
  ├─ Ruído de Rayleigh range-dependente + streak central (ray_noise_param, central_peak, central_std)
  ├─ intensity_offset + intensity_gain: pós-processamento após normalização
  │
  └─ make_sonar_map_*: (r, azi) → (r·cos(azi), r·sin(azi), intensity) — Cartesiano não usado na imagem
  │
  ▼ make_sonar_image()   [kernel Warp]
  sonar_image[i, width-j] = intensity  ← eixo de azimute INVERTIDO (off-by-one em j=0)
  resultado: grid (N_range × N_azi × 4) RGBA uint8
    Linha i    = range bin i  (i=0 = min_range, i=N-1 = max_range)
    Coluna k   = azimute bin (n_azi-1-k) invertido pelo kernel
                 → coluna 0 ≈ azi_max=155° (ESQUERDA)
                 → coluna N-1 ≈ azi_min=25° (DIREITA)
  │
  ▼ OceanSimImagingSonar.update_state()  [sensors.py]
        rgb_image.set_value(sonar_np[...,:3])  → Buffer com tag "rgb"
```

### Convenção de azimute (crítica para a projeção Cartesiana)

| Posição real | Azimute | cos(azi) | x_cart | Lado na imagem polar |
|---|---|---|---|---|
| Frente (boresight) | 90° | 0 | 0 | Centro vertical |
| Direita | < 90° | > 0 | positivo | Direita |
| Esquerda | > 90° | < 0 | negativo | Esquerda |

A rotação de eixos em `world2local`:
```python
pcl_local = (camera_X, -camera_Z, camera_Y)
# camera_X = lateral direita do sonar
# -camera_Z = frente do sonar (câmera USD olha em -Z)
```

---

## 3. Parâmetros de `make_sonar_data()` — tabela completa

Todos os parâmetros abaixo são configuráveis em tempo real via UI ou YAML.

| Parâmetro | Padrão PatoSim | Range útil | Efeito |
|---|---|---|---|
| `attenuation` | 0.3 | 0.0 – 2.0 | Atenuação por distância: `intensity *= exp(-a*dist)`. Maior = objetos distantes desaparecem mais rápido |
| `gau_noise_param` | 0.05 | 0.0 – 1.0 | Amplitude do ruído gaussiano multiplicativo: `intensity *= (0.5 + N(0, σ))`. Aumentar = textura granular na imagem |
| `ray_noise_param` | 0.05 | 0.0 – 1.0 | Escala do ruído de Rayleigh: cresce com o range. Simula reverberação de fundo dependente da distância |
| `intensity_offset` | 0.0 | -1.0 – 1.0 | Offset somado após normalização. Valores positivos iluminam objetos fracos; negativos suprimem fundo |
| `intensity_gain` | 1.0 | 0.1 – 5.0 | Ganho multiplicado após normalização. Aumentar = contraste global maior |
| `central_peak` | 2.0 | 0.0 – 10.0 | Intensidade do streak central (artefato de boresight do sonar real). 0 = desativado |
| `central_std` | 0.001 | 0.0001 – 0.05 | Largura angular do streak (rad²). Maior = streak mais largo e suave |
| `binning_method` | "sum" | "sum" \| "mean" | Acumulação por bin: "sum" amplifica retornos sobrepostos; "mean" suaviza. Ajustar `gau_noise_param` ao trocar |
| `normalizing_method` | "range" | "all" \| "range" | Normalização: "range" por faixa de distância (preserva variação lateral); "all" por máximo global (maior contraste absoluto) |

### Streak central

O streak central é o artefato luminoso na linha de boresight (azi=90°) presente em sonares
reais causado pela maior densidade de energia no feixe central. É modelado pelo kernel
`range_dependent_rayleigh_2d` com fator:

```
noise[i,j] *= (r[i,j]/max_range)² × (1 + central_peak × exp(-(azi[i,j] - π/2)² / central_std))
```

---

## 4. Integração com o PatoSim (sensors.py)

**Arquivo:** `exts/omni.ext.patosim/omni/ext/patosim/sensors.py`

### Classe `OceanSimImagingSonar`

```python
class OceanSimImagingSonar(Sensor):
    rgb_image  = Buffer(tags=["rgb"])       # grid r×azi RGBA — saída do make_sonar_image()
    pointcloud = Buffer(tags=["pointcloud"]) # PCL bruto do annotator
    position   = Buffer()
    orientation = Buffer()
    status     = Buffer("idle")
```

**Estado de renderização** (todos configuráveis via `set_render_model_params()` ou YAML):
```python
_attenuation       = 0.3
_gau_noise_param   = 0.05
_ray_noise_param   = 0.05
_intensity_offset  = 0.0
_intensity_gain    = 1.0
_central_peak      = 2.0
_central_std       = 0.001
_binning_method    = "sum"
_normalizing_method = "range"
```

**Métodos de parâmetros:**
- `set_render_model_params(**kwargs)` — atualiza qualquer subconjunto de parâmetros em runtime
- `get_params_as_dict()` → `dict` — exporta todos os parâmetros atuais
- `load_params_from_yaml(path)` → `bool` — carrega do arquivo YAML (esquema `sonar.render.*`)
- `save_params_to_yaml(path)` → `bool` — salva no arquivo YAML

**Métodos de preview:**
- `render_polar_preview(size=400)` → `np.ndarray (size, size, 4)` RGBA
  Projeção Cartesiana fan: `row ∝ 1 - r·sin(azi)/max_r`, `col ∝ r·cos(azi)`
  Inclui overlay de arcos de distância e linhas de ângulo via cv2.
- `render_planar_preview(width, height)` → `np.ndarray (height, width, 4)` RGBA
  Grid r×azi bruto redimensionado (projeção nativa OceanSim) com colormap acústico.

---

## 5. Integração com o sistema de gravação (MobilityGen/PatoSim)

### Fase online (gravação em tempo real)

O sonar **não** é gravado durante a fase online de navegação porque
`deferred_sensor_processing_enabled = True` (padrão). O buffer `rgb_image` existe mas
`state_dict_rgb()` não é chamado online.

Gravado online: `state/common/*.npy` — pose do sonar (`position`, `orientation`).

### Fase offline (replay)

O sonar é capturado como parte do replay de câmeras:

1. `replay_implementation.py` → `scenario.write_replay_data()` → `robot.update_state()` → `sonar.update_state()`
2. `writer.write_state_dict_rgb(scenario.state_dict_rgb(), step)` captura todos os buffers
   com tag `"rgb"`, **incluindo** `sonar.rgb_image`
3. Salvo em: `state/rgb/sonar/XXXXXXXX.jpg` — grid bruto `(N_range × N_azi)` = 1960×520

**Importante:** a resolução gravada é determinada pelos parâmetros físicos do sensor
(`range_res`, `angular_res`), **não** pelas configurações de preview da UI.

### Diferença em relação ao LiDAR

| | Sonar | LiDAR |
|---|---|---|
| Captura online | Não (deferred) | Não (deferred) |
| Replay: módulo separado | Não (implícito via `state_dict_rgb`) | Sim (`drive_rtx_lidar_render_products`) |
| Gravado via | `state_dict_rgb()` → JPEG | `state_dict_pointcloud()` → PLY/NPY |
| Formato de saída | JPEG 1960×520 (grid r×azi) | PLY / NPY (nuvem de pontos) |
| Warm-up | Não necessário | Sim (2–3 frames RTX) |

---

## 6. Parâmetros configuráveis em `OceanSimROVRobot` (robots.py)

```python
sonar_translation          = (0.3, 0.0, 0.3)      # posição relativa ao ROV (m)
sonar_orientation_euler_deg = (0.0, 0.0, 0.0)      # orientação (graus)
sonar_max_range            = 10.0                   # alcance máximo (m)
enable_sonar               = True                   # habilitar/desabilitar sensor
```

Os demais parâmetros físicos (`hori_fov`, `angular_res`, `range_res`) são definidos
em `OceanSimImagingSonar.build()` e espelham o hardware Oculus M370s. Alterá-los
exige reconstruir o sensor — não são alteráveis em runtime.

---

## 7. Preview na UI

### 7.1 Janela Sensor Preview (Camera Preview)

Exibe **apenas** câmeras ópticas (Front Camera + Underwater Camera). O sonar foi
removido deste painel — foi movido para a janela dedicada Sonar Preview.

Seletor de resolução com 4 presets:

| Preset | Câmeras | Descrição |
|---|---|---|
| Pequeno | 200×113 | Baixo consumo, painel compacto |
| Médio | 256×144 | Padrão (equivalente ao antigo "simplified") |
| Grande | 320×180 | Alta qualidade (equivalente ao antigo "robust") |
| HD | 426×240 | Máxima resolução de preview |

### 7.2 Janela Sonar Preview (dedicada) — `PatoSim - Sonar Preview`

Janela flutuante acessível via toggle "Preview Sonar" no Window Manager.
Exibe duas vistas **lado a lado**:

```
┌──────────────────────────────────────────────────────┐
│  Fan polar (Cartesiano)    Grid r×azi (OceanSim)     │
│  ┌──────────┐              ┌───────┐                  │
│  │          │              │  pw   │                  │
│  │  sz × sz │              │  ×    │                  │
│  │   fan    │              │  sz   │                  │
│  │          │              │       │                  │
│  └──────────┘              └───────┘                  │
│  [Altura px: 400] [Aplicar]                           │
│  ▶ Advanced                                           │
└──────────────────────────────────────────────────────┘
```

Onde `sz` = tamanho configurável (padrão 400) e `pw = round(sz × 0.2653)` ≈ 106 px para sz=400.

**Vista Fan polar (esquerda):**
- Projeção Cartesiana bird's-eye: frente = topo, sensor = base, laterais = lados
- Overlay de arcos de distância (2, 4, 6, 8, 10 m) e linhas angulares (±15°, ±30°, ±45°, ±60°)
- Colormap acústico: azul→ciano→verde→amarelo→branco por intensidade

**Vista Grid r×azi (direita):**
- Grid bruto OceanSim redimensionado: linha = range, coluna = azimute (invertido pelo kernel)
- Proporção natural 1:3.77 respeitada — mesma largura de pw pixels, altura sz pixels
- Ideal para diagnóstico: verificar preenchimento de bins, artefatos de ruído, streak central

### 7.3 Seção Advanced (parâmetros em tempo real)

| Grupo | Controle | Tipo |
|---|---|---|
| Ruído | Gau Noise, Ray Noise, Attenuation | FloatDrag |
| Streak | Central Peak, Central Std | FloatDrag |
| Intensidade | Offset, Gain | FloatDrag |
| Métodos | Binning (sum/mean), Normalize (all/range) | ComboBox |
| YAML | Path, [Load], [Save] | StringField + Buttons |

**Fluxo YAML:**
1. Digitar ou colar o caminho do arquivo `.yaml` no campo Path
2. Clicar **Load** → `sonar.load_params_from_yaml(path)` → UI atualiza automaticamente
3. Clicar **Save** → aplica valores atuais da UI → `sonar.save_params_to_yaml(path)`

---

## 8. Diagnóstico e problemas comuns

### Sonar não produz imagem

1. `enable_sonar = False` em `OceanSimROVRobot` → sonar não é instanciado
2. `enable_rgb_rendering()` não chamado → `_rgb_enabled = False`
3. Cena sem geometria no FOV → `semanticSeg_annot.get_data()` vazio → `scan()` retorna `False`
4. Reflectividade não aplicada → intensidade nula → imagem toda preta

### Imagem virada horizontalmente (grid bruto)

Causa: o kernel escreve `sonar_image[i, width-j]`, invertendo o eixo de azimute.
O grid bruto já tem a orientação correta (esquerda = esquerda) após o flip do kernel.
Na projeção fan, o linspace de azimute é invertido (`azi_deg_max → azi_deg_min`) para compensar.

### Imagem distorcida no Camera Preview

Causa histórica: o grid bruto (1960×520, proporção 3.77:1) era exibido em slots 16:9.
Solução atual: sonar removido do Camera Preview — exibido somente na janela Sonar Preview
com proporção natural respeitada.

### Streak central muito intenso

Reduzir `central_peak` (padrão 2.0). Para desativar completamente: `central_peak = 0.0`.

### Sonar não salvo no replay

Verificar que `rgb_enabled` está ativo no script de replay (`--rgb`).
O sonar é capturado junto com as câmeras em `state/rgb/sonar/`.

---

## 9. Melhorias implementadas (histórico)

| Versão | Melhoria |
|---|---|
| v1 | Wrapper `OceanSimImagingSonar` com buffers de preview e gravação |
| v2 | `render_polar_preview()` — projeção Cartesiana fan com colormap acústico e overlay de escala |
| v2 | `render_planar_preview()` — grid bruto redimensionado com colormap |
| v2 | Janela flutuante Sonar Preview com modo polar/planar |
| v3 | Correção da inversão horizontal (linspace revertido para compensar kernel `width-j`) |
| v3 | Correção da rotação 90° (x_cart↔y_cart trocados no mapeamento de pixels) |
| v3 | ComboBox modo polar/planar corrigido (`add_item_changed_fn`) |
| v4 | Vista dual lado a lado: fan polar + grid OceanSim nativo na mesma janela |
| v4 | Sonar removido do Camera Preview — seletor de resolução 4 presets para câmeras |
| v4 | Advanced: todos os parâmetros de `make_sonar_data()` expostos na UI |
| v4 | Suporte a YAML: `load_params_from_yaml()` / `save_params_to_yaml()` |

## 10. Melhorias propostas (pendentes)

| Prioridade | Melhoria |
|---|---|
| Média | Salvar imagem polar (fan) no replay além do grid bruto (`state/rgb/sonar_polar/`) |
| Média | Reconhecer `OceanSimImagingSonar` como módulo de câmera em `_is_camera_module()` para controle independente no replay |
| Baixa | Buffer `npy` separado para `sonar_map` (float32) — análise quantitativa de intensidade |
| Baixa | Decimação assíncrona: sonar real opera a 40 Hz, PatoSim processa a 100 Hz |
| Baixa | Multi-beam 3D: combinar PCL do sonar com pose 6-DOF para SLAM acústico |

---

## 11. Referências de código

| Arquivo | Conteúdo |
|---|---|
| `isaacsim/oceansim/sensors/ImagingSonarSensor.py` | Backend: Camera + Warp, `make_sonar_data()`, `scan()` |
| `isaacsim/oceansim/utils/ImagingSonar_kernels.py` | Kernels GPU: `world2local`, `bin_intensity`, `make_sonar_map_*`, `make_sonar_image` |
| `exts/omni.ext.patosim/omni/ext/patosim/sensors.py` | Wrapper: `OceanSimImagingSonar`, parâmetros, preview, YAML |
| `exts/omni.ext.patosim/omni/ext/patosim/robots.py` | Configuração e instanciação do sonar no ROV |
| `exts/omni.ext.patosim/omni/ext/patosim/extension.py` | UI: Sonar Preview dual, Advanced params, Camera Preview presets |
| `docs/codex_prompt_sonar_advanced_params.md` | Prompt Codex: Advanced params + YAML |
| `docs/codex_prompt_sonar_preview_dual_view.md` | Prompt Codex: vista dual fan + grid |
| `docs/codex_prompt_sonar_preview_fixes.md` | Prompt Codex: correções de orientação e modo planar |
| `docs/codex_prompt_sonar_improvements.md` | Prompt Codex: melhorias acústicas futuras |
| `scripts/replay_implementation.py` | Captura offline via `state_dict_rgb()` |
| `exts/omni.ext.patosim/omni/ext/patosim/writer.py` | Persistência: `write_state_dict_rgb()` → JPEG |
