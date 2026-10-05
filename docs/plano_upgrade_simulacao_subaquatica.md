# Plano de avaliação e upgrade da simulação subaquática (PatoSim / OceanSim)

Documento de planejamento — **nenhuma alteração de código foi feita**. Este documento
descreve o estado atual (com evidências concretas de arquivo/linha), causas-raiz dos
problemas relatados, e um plano de implementação e testes faseado, para revisão antes
de qualquer execução.

Repositórios avaliados:
- `MOD_patosim` (este repo) — extensão `exts/omni.ext.patosim/omni/ext/patosim/`
- `~/Documents/Documentos/BEVLOG/bevlog-isaac/MobilityGen` — fork de referência com
  pipeline de gravação de dados mais madura/funcional, usado como base de comparação

---

## 1. Resumo executivo

O MOD_patosim é um fork do NVIDIA MobilityGen (renomeado "PatoSim") integrado ao
OceanSim para simulação subaquática. A arquitetura de gravação, sensores e cena já foi
majoritariamente portada — **muito mais do que o TODO do README sugere**. Os problemas
relatados têm causas-raiz pontuais e já identificadas no código, não exigem reescrever
a arquitetura:

| Problema relatado | Causa-raiz confirmada | Complexidade do fix |
|---|---|---|
| Gravação de dados não funcional | Múltiplos gaps pontuais na gravação *ao vivo* (sonar bruto nunca persistido, falhas silenciosas) — replay já funciona melhor que o live recording | Média |
| Assets sem textura / fora de escala | `asset_path.json` canônico aponta para um caminho inválido de outra máquina; nenhuma normalização de unidades (`metersPerUnit`) ao inserir assets | Baixa |
| Sonar não varia com distância | Normalização por anel de alcance (`normalizing_method="range"`) apaga o falloff exponencial já calculado corretamente no kernel | Baixa |
| Mapa de ocupação é 2D | Confirmado: é uma única fatia XY em `rov_operating_depth ± occupancy_map_z_half` | Média/Alta (para 3D completo) |
| Falta lidar subaquático | Existe uma classe `Lidar` genérica (terrestre, desligada por padrão), sem perfil/atenuação subaquática | Média |
| Falta joystick para o ROV | Driver de gamepad já portado (`inputs.py`) mas nunca conectado a um `Scenario` do ROV | Baixa |
| Falta navegação autônoma básica | **Já existe** (`OceanSimROVWaypointScenario`, `OceanSimROVPathFollowingScenario`) — precisa validação e testes, não implementação do zero | Baixa (validação) |
| Pose relativa dos sensores | Pose (mundo) de robô e de cada sensor **já é gravada a cada passo** — falta apenas o extrínseco relativo fixo e a documentação do schema | Baixa |

---

## 2. Estado atual por subsistema (evidências)

### 2.1 Gravação de dados

Arquitetura: `common.py` (árvore `Module`/`Buffer`, idêntica ao MobilityGen original),
`extension.py` (laço de gravação ao vivo), `writer.py` (persistência), `scripts/replay_implementation.py`
(renderização offline).

- Pose do robô e de **todos** os sensores é gravada: cada classe de sensor declara
  `self.position = Buffer()` / `self.orientation = Buffer()` sem tag, então cai em
  `state/common/*.npy` a cada passo (`sensors.py:339-340` câmera, `502-503` lidar,
  `974-975` câmera UW, `1423-1424` sonar, `2033-2034` DVL, `2130-2131` barômetro).
- DVL e barômetro **já são gravados corretamente** (buffers propositalmente sem tag —
  comentário em `sensors.py:2029-2035` e `:2128`). Este item do TODO do README já está feito.
- **Sonar bruto tem gap real na gravação ao vivo**: o buffer `sonar_intensity`
  (`sensors.py:1421`, tag `"sonar_intensity"`) é calculado todo frame (`sensors.py:1724`)
  mas não bate com nenhum `state_dict_*` da gravação ao vivo — é descartado.
  `Writer.write_sonar_data_package` (`writer.py:91-160`, gera .npy bruto / PNG16 / PNG
  polar / metadados JSON) **nunca é chamado a partir de `extension.py`** durante gravação
  ao vivo — só é chamado no replay offline (`scripts/replay_implementation.py:373-394`),
  atrás de flags de CLI que default para `False`. O `Config` já tem os campos certos
  (`sonar_save_raw_npy=True`, `sonar_save_png16=False`, `sonar_save_polar_png=False`,
  `config.py:29-31`) mas eles não são usados durante a gravação ao vivo.
- `Writer.write_state_dict_rgb` é chamado em `extension.py:3340` **sem o argumento
  `sonar_ref`**, então o branch de preview polar do sonar (`writer.py:65-89`) nunca
  dispara ao vivo.
- Todos os `write_state_dict_*` de sensores no laço ao vivo estão envoltos em
  `except: pass` (`extension.py:3338-3360`) — qualquer falha de sensor é **silenciosa**.
  Isso é o principal suspeito para "gravação não funcional": provavelmente há erros reais
  sendo engolidos.
- Câmera estéreo existe (`OceanSimStereoUWCamera`, `sensors.py:1332`) mas vem **desligada
  por padrão** (`enable_rov_stereo_camera: bool = False`, `config.py:20`).
- **Câmera fisheye/esférica não existe.** Há referências defensivas
  (`getattr(robot, "fisheye_left", None)`) espalhadas em `extension.py` (linhas 1024,
  2746-2756, 2799-2800, 4014, 4039), mas nenhuma classe `Robot` define esses atributos e
  nenhuma classe de sensor fisheye existe em `sensors.py`. É UI morta.

### 2.2 Sonar — por que não varia (muito) com a distância

- Por ponto, o kernel já calcula atenuação exponencial correta:
  `intensity = reflectivity * cos_theta * exp(-attenuation * dist)`
  (`oceansim/utils/ImagingSonar_kernels.py:45`), com `attenuation=0.3`
  (`sensors.py:1406`, documentado em `docs/subsections/sonar_pipeline.md`).
- **O problema é o pós-processamento**: `normalizing_method` default é `"range"`
  (`sensors.py:1412`). Em `make_sonar_map_range`
  (`ImagingSonar_kernels.py:178-201`), cada anel de alcance é dividido pelo **próprio
  máximo daquele anel** (`ImagingSonar_kernels.py:190-191`, máximo por linha calculado no
  kernel `range_max`, linhas 106-109). Isso reescala cada faixa de distância
  independentemente para preencher 0–1 — um alvo a 0.3 m e um alvo idêntico a 9 m acabam
  com brilho de pico comparável, porque cada anel é normalizado na própria escala.
- `docs/subsections/sonar_pipeline.md` já documenta isso: `"range"` "preserva variação
  lateral" vs `"all"` (máximo global do frame) que dá "maior contraste absoluto" — ou
  seja, o comportamento observado é o modo default fazendo exatamente o que foi
  projetado para fazer, só que esse não é o modo certo para geração de dataset de ML.
- **Implicação para o projeto**: os modelos de fusão sensorial do usuário (`DPS_sonar_net`,
  `PSNetSonarPriority`) usam o range/intensidade do sonar como sinal de treino — sonar
  achatado por distância destrói exatamente a informação que esses modelos precisam.

### 2.3 Assets — textura e escala

- Carregamento da cena principal é um `add_reference_to_stage` simples
  (`build.py:433,493`), sem normalização de `metersPerUnit`/`scale` em lugar nenhum do
  código (`grep` por `MetersPerUnit`/`SetScale`/`scale_to_meters` = zero ocorrências).
- Inserção de objetos de dataset (`build.py:351-420+`) usa `scale = config.dataset_object_scale`,
  **default fixo em `1.0`** (`config.py:27`), sem detectar a unidade nativa do asset de
  origem. Assets de scan/fotogrametria costumam ser autorados em cm — sem compensação,
  isso produz exatamente "fora de escala".
- **Causa raiz confirmada e verificada nesta máquina**: existem dois `asset_path.json`
  conflitantes.
  - `isaacsim/oceansim/utils/asset_path.json` (fora do layout documentado) →
    `/media/external_01/OceanSim_assets` — **este caminho não existe nesta máquina**
    (`/media/external_01` não existe).
  - `exts/omni.ext.patosim/omni/ext/patosim/oceansim/utils/asset_path.json` (o caminho
    canônico, lido por `assets_utils.get_oceansim_assets_path()`) →
    `/home/bevlog/PatoSIM/scripts/launch_patosim.sh` — resíduo de outra máquina/usuário,
    nem é um diretório.
  - **O caminho real e válido nesta máquina é `/mnt/external/isaac/OceanSim_assets`**
    (contém `Bluerov/`, `collected_MHL/`, `collected_rock/` — confirmado via `ls`).
  - `get_oceansim_assets_path()` (`assets_utils.py:49-53`) lança `FileNotFoundError`
    quando o caminho resolvido não é diretório. `robots.py:521-524` captura essa exceção
    e cai para `assets/models/` local do repo — por isso o BlueROV ainda carrega, mas
    qualquer asset que dependa da raiz *registrada* do OceanSim fica quebrado (texturas/
    modelos ausentes). `oceansim/modules/SensorExample_python/ui_builder.py:292,302,321`
    chama `get_oceansim_assets_path()` sem fallback — quebra se esse módulo for usado.
- Nenhuma revalidação de materiais/texturas é feita após `add_reference_to_stage` — se um
  asset referenciar texturas por caminho relativo e for referenciado num contexto Nucleus/
  layer diferente, a textura falha silenciosamente e o objeto some renderizado cinza/branco.

### 2.4 Mapa de ocupação (2D confirmado)

- `OccupancyMap` (`occupancy_map.py:55-507`) é estritamente 2D: array `(H, W)`, resolução
  escalar, origem `(x, y, yaw)`. Não existe eixo Z em nenhum método.
- Geração para o ROV (`build.py:512-529` → `build.py:87-111`) usa **uma única fatia**:
  `z_min = rov_operating_depth - occupancy_map_z_half`,
  `z_max = rov_operating_depth + occupancy_map_z_half`
  (default `rov_operating_depth=-2.0`, `occupancy_map_z_half=3.0`, `config.py:26-27`) —
  ou seja, uma banda fixa de ±3 m centrada numa única profundidade "nominal".
  Qualquer falha cai silenciosamente num placeholder 256×256 totalmente livre
  (`build.py:73-84,111`) — **sem nenhuma célula ocupada**, o que mascara erros de geração.
- `OceanSimROVRobot.occupancy_map_z_min/z_max` (`robots.py:400-401`, `-5.0`/`5.0`) são
  **código morto** para o caminho subaquático — só o branch de robô terrestre
  (`build.py:530-536`) os usa.
- Para um veículo que se move tanto em Z quanto em XY (ROV, não robô terrestre), uma
  única fatia 2D esconde obstáculos acima/abaixo da banda escolhida.

### 2.5 Navegação e controle do robô

- 3 cenários **registrados** e funcionais para o ROV (`scenarios.py:150,207,284`):
  `OceanSimROVTeleoperationScenario` (teclado, 6DOF real), `OceanSimROVWaypointScenario`,
  `OceanSimROVPathFollowingScenario` — **a navegação autônoma básica pedida já existe**.
- Todos os cenários antigos de robô terrestre do MobilityGen (`KeyboardTeleoperationScenario`,
  `GamepadTeleoperationScenario`, `RandomAccelerationScenario`, `RandomPathFollowingScenario`
  e variantes de empilhadeira) continuam no arquivo mas com `@SCENARIOS.register()`
  **comentado** (linhas 498, 1005, 1112, 1225, 1346, 1384, 1415, 1461), com comentário
  explícito (`scenarios.py:145-147`): mantidos só como referência, pois assumem mapa 2D
  e dinâmica de veículo terrestre. **Não existe hoje nenhum cenário de gamepad para o ROV.**
- `OceanSimROVRobot` (`robots.py:385-865`) tem física 6DOF real e não-trivial em
  `underwater_physics.py` (`BlueROVUnderwaterPhysics`): matriz de alocação de 8
  propulsores, mixagem por pseudo-inversa, atraso de primeira ordem por propulsor, arrasto
  linear+quadrático por eixo, empuxo dependente de fração submersa. Parâmetros expostos:
  `linear_drag_coeffs`, `quadratic_drag_coeffs`, `angular_drag_coeffs`,
  `angular_quadratic_drag_coeffs`, `thruster_max_force_newtons`,
  `thruster_time_constant_s`, `center_of_buoyancy_body_m` (`underwater_physics.py:26-39`).
  Isso **não** é diferencial-drive — já é controle de ROV de verdade.
- **Já existe um `GamepadDriver` funcional em `inputs.py`** (`GamepadAxis` linha 169,
  `GamepadDriver` linha 199, `get_axis_values()` linha 268), portado do MobilityGen via
  `carb.input`, mas **nunca conectado a nenhum `Scenario` de ROV**. Portar o mapeamento
  6DOF já existente em `_ROVKeyboardController` (`scenarios.py:67-106`) para ler eixos do
  gamepad em vez do teclado é um trabalho pequeno e de baixo risco.
- Pontos de montagem dos sensores (sonar, DVL, câmeras, lidar) são atributos de classe
  centralizados em `OceanSimROVRobot` (`robots.py:444-451`) e aplicados em `build()`
  (`robots.py:626-702`) — já é um único lugar de calibração, ao contrário do que o TODO
  do README sugere (esse item parece desatualizado).

---

## 3. Plano de implementação (faseado)

Ordem pensada para priorizar correções de baixo risco e alto impacto primeiro, e deixar
a mudança arquitetural mais arriscada (mapa de ocupação 3D) por último — cada fase é
razoavelmente independente e testável isoladamente.

### Fase 0 — Correções urgentes e de baixo risco

1. **Corrigir registro de assets**: reexecutar
   `scripts/register_oceansim_assets.sh /mnt/external/isaac/OceanSim_assets` (caminho
   real confirmado nesta máquina) para sobrescrever os dois `asset_path.json` inválidos.
   Adicionar validação no startup (mensagem clara na UI, não uma exceção silenciosa
   engolida) quando o caminho registrado não existir.
2. **Corrigir normalização do sonar**: expor `normalizing_method` como modo explícito no
   `Config`/UI com pelo menos três presets:
   - `raw` (sem normalização por anel — preserva o falloff exponencial bruto, melhor
     para treinar `DPS_sonar_net`/`PSNetSonarPriority`)
   - `range` (comportamento atual — bom para inspeção visual humana / contraste lateral)
   - `all` (normalização global do frame — preserva contraste de distância)
   Trocar o **default usado na geração de dataset** para `raw` ou `all`; manter `range`
   disponível como preset de "preview/visualização".
3. **Ligar a persistência de sonar bruto na gravação ao vivo**: chamar
   `writer.write_sonar_data_package(...)` também no laço `on_physics` (não só no
   replay), usando as flags já existentes em `Config` (`sonar_save_raw_npy` etc.).
   Corrigir a chamada de `write_state_dict_rgb` em `extension.py:3340` para passar
   `sonar_ref`.
4. **Remover `except: pass` silenciosos** no laço de gravação de sensores
   (`extension.py:3338-3360`) — logar o erro real (nome do sensor + exceção). Isso por
   si só pode revelar a causa concreta de "gravação não funcional".
5. **Compensar unidade de escala ao inserir assets**: em vez do multiplicador fixo
   `config.dataset_object_scale`, calcular
   `effective_scale = config.dataset_object_scale * (metersPerUnit_do_asset / metersPerUnit_do_stage)`
   lendo `UsdGeom.GetStageMetersPerUnit` do layer do asset referenciado antes de aplicar
   a escala.

### Fase 1 — Pose, extrínsecos e schema de dados

1. Manter a gravação por-frame de pose (mundo) de robô e sensores, que já funciona.
2. Adicionar um `sensors_extrinsics.json` estático (gravado uma vez por sessão, junto de
   `occupancy_map/`, `config.json`, `stage.usd`) com a transformação corpo-sensor fixa de
   cada sensor, lida diretamente das constantes de montagem já centralizadas em
   `robots.py:444-451`. Isso complementa (não substitui) a pose absoluta por frame, e
   evita que consumidores downstream tenham que recompor manualmente
   `T_world_sensor = T_world_robot * T_robot_sensor` por composição de quaternions.
3. Documentar formalmente o schema de pose no README/`docs/`: convenção de eixos, ordem
   dos componentes do quaternion (`wxyz` vs `xyzw`), unidades (metros), referencial
   (mundo Isaac Sim) — hoje isso não está escrito em lugar nenhum e é uma fonte clássica
   de bug silencioso em pipelines de SfM/3DGS a jusante.

### Fase 2 — Lidar subaquático

Ver seção 4 para a análise do sensor de referência. Resumo da implementação:

1. Criar 1–2 perfis de `LidarRtx` com alcance e abertura drasticamente reduzidos frente
   ao perfil terrestre atual (que usa os defaults longos do RTX Lidar).
2. Adicionar um pós-processamento de atenuação/queda de retorno análogo ao do sonar
   (mesma lei exponencial tipo Beer-Lambert, mas com coeficiente calibrado para laser
   azul-verde em água, não para som), incluindo probabilidade de retorno válido caindo
   com a distância — para não gerar nuvens de pontos "limpas demais" e irrealistas.
3. Habilitar por padrão no `OceanSimROVRobot` (hoje `enable_rov_lidar=False`) e validar
   a gravação ponta-a-ponta (o `pointcloud` já está com a tag certa e pose já é gravada,
   então não deve exigir mudança de schema).
4. Investigar durante a implementação se `water_profile_path` (mecanismo já existente,
   ainda não inspecionado a fundo) pode alimentar um coeficiente de turbidez único
   compartilhado entre sonar e lidar, em vez de duplicar o conceito.

### Fase 3 — Navegação: joystick e validação da navegação autônoma

1. Implementar `OceanSimROVGamepadTeleoperationScenario`, espelhando
   `OceanSimROVTeleoperationScenario`/`_ROVKeyboardController` mas lendo
   `GamepadDriver.ensure_connected().get_axis_values()` em vez do teclado. Mapeamento
   sugerido (convenção comum em joysticks de ROV comerciais, compatível com os 4 eixos
   já expostos por `GamepadAxis`): stick esquerdo → surge/sway (frente-trás/lateral),
   stick direito → yaw/heave (giro/subida-descida). Roll/pitch ficam de fora do controle
   manual, coerente com a estabilidade passiva já modelada via
   `center_of_buoyancy_body_m` deslocado do centro de massa.
2. Adicionar ganhos e curva de resposta configuráveis (separados dos do teclado):
   `rov_gamepad_linear_gain`, `rov_gamepad_vertical_gain`, `rov_gamepad_angular_gain`,
   deadzone, e uma curva de expoente (`expo`) para permitir precisão fina em baixa
   deflexão do stick — prática comum em pilotagem de ROV.
3. Zerar a ação automaticamente se o controle desconectar no meio de uma sessão
   (checagem de segurança, aproveitando `ensure_connected()` já existente).
4. **Validar** (não reimplementar) `OceanSimROVWaypointScenario` e
   `OceanSimROVPathFollowingScenario` como a "navegação autônoma básica": confirmar que
   operam de forma confiável, checar se atualmente só navegam no plano XY em profundidade
   fixa (consequência do mapa de ocupação 2D atual — a confirmar), e documentar as
   limitações até a Fase 4 estender o mapa de ocupação.
5. Expor no `Config`/UI os parâmetros de `underwater_physics.py` que hoje só existem como
   defaults de dataclass (`linear_drag_coeffs`, `thruster_max_force_newtons`,
   `thruster_time_constant_s`, etc.) para permitir ajuste de "sensação" de pilotagem sem
   editar código.

### Fase 4 — Mapa de ocupação: de 2D para consciente de profundidade

Duas sub-fases, para não travar o projeto numa reescrita arriscada de uma vez:

**Fase 4a (baixo risco, reaproveita a máquina 2D existente)**: gerar uma pilha de N
fatias 2D (`OccupancyMapStack`) cobrindo toda a faixa operacional de profundidade do
ROV, em vez de uma única banda `rov_operating_depth ± occupancy_map_z_half`. Os
cenários de navegação passam a consultar a fatia correspondente à profundidade atual (ou
checar fatias vizinhas antes de subir/descer). Reaproveita 100% do gerador de omap do
Isaac Sim (`isaacsim.asset.gen.omap`) já usado hoje, só chamado N vezes.

**Fase 4b (maior esforço, mapa realmente 3D)**: generalizar `OccupancyMap` para um grid
voxel 3D — seja empilhando as fatias da Fase 4a num array `(Z, H, W)`, seja generalizando
o rasterizador de fallback já existente (`occupancy_map_utils._rasterize_visible_geometry_fallback`,
hoje baseado em bounding boxes 2D) para voxels. Requer:
- Generalizar `world_to_pixel`/`pixel_to_world` → `world_to_voxel`/`voxel_to_world`.
- Generalizar `buffered()` (hoje dilatação 2D via OpenCV) → dilatação morfológica 3D.
- Planejamento de caminho 3D para `OceanSimROVPathFollowingScenario` (o planejador C++
  atual do MobilityGen é 2D; provavelmente é necessário um A*/Dijkstra 3D em grid,
  inicialmente em Python puro, com resolução mais grossa — ex.: 0.25–0.5 m — para manter
  a busca tratável).
- Formato de exportação: o formato ROS (`map.png`+`map.yaml`) é inerentemente 2D; propor
  um formato adicional (`.npy` do voxel grid + `.yaml` com resolução/origem/shape) em vez
  de forçar o formato ROS a representar 3D.

Recomenda-se manter a Fase 4a como modo padrão (rápido, suficiente para operação perto de
uma profundidade nominal) e a Fase 4b como modo opcional (`occupancy_map_mode:
"2d_band" | "3d_voxel"`) para missões com grande variação vertical (ex.: inspeção de
naufrágio/estrutura).

### Fase 5 — Varredura ampla de assets e mundos

Auditoria sistemática (não só correção pontual): validar textura/material e escala de
cada asset em `assets/models/{plataforms,statues_temples,scenario}` (as pastas
permitidas por `build.py:39-40`), documentando quais mundos/assets são oficialmente
suportados vs. de teste — item que o próprio TODO do README já pedia
("Worlds / scene usage", "Default assets folder").

---

## 4. Lidar subaquático — sensor de referência e proposta

Lidar óptico de tempo-de-voo convencional (o mesmo princípio usado em terra) tem alcance
efetivo muito curto debaixo d'água por causa da absorção e do backscatter da luz — por
isso veículos subaquáticos reais não usam "lidar" no sentido terrestre, e sim
**scanners a laser de triangulação/luz estruturada** de curto-médio alcance. O análogo
comercial mais próximo hoje, usado em ROVs/AUVs para reconstrução 3D (exatamente o caso
de uso deste projeto), é a linha **Voyis Insight**:

| Modelo | Alcance | Profundidade máx. | Perfil de uso |
|---|---|---|---|
| Insight Nano | 0.13–2.5 m | 1000 m | portátil/diver, curto alcance |
| **Insight Micro** | **até 7 m** | 1000 m | **ROV/AUV pequeno — recomendado como perfil primário** (compatível com a escala do BlueROV já modelado) |
| Insight Pro | 1.5–15 m | 6000 m | maior alcance, veículos maiores |

Fonte: [Voyis — Insight Pro](https://www.unmannedsystemstechnology.com/company/voyis/insight-pro-underwater-laser-scanner/),
[Voyis Insight Laser Scanners](https://voyis.com/insight-laser-scanners/),
[Unmanned Systems Technology — Voyis](https://www.unmannedsystemstechnology.com/company/voyis/).

**Proposta**: calibrar o perfil sintético de lidar em cima do envelope do **Insight
Micro** (alcance ~7 m) como padrão, por ser o mais compatível com a escala de um ROV
pequeno tipo BlueROV, com um segundo preset opcional no envelope do **Insight Pro**
(~15 m) para cenários de água mais clara. Isso é uma mudança bem mais realista do que
simplesmente reduzir o alcance de um perfil de lidar terrestre genérico "no chute".

Tecnicamente, a classe `Lidar` já existente (`sensors.py:470+`, já usa `LidarRtx` do
Isaac Sim com fallback via Replicator) é reaproveitável como base — falta apenas (a) um
perfil de configuração com alcance/abertura realistas, (b) um modelo de atenuação/queda
de retorno com a distância análogo ao já implementado para o sonar, e (c) ligar por
padrão no robô ROV.

---

## 5. Plano de testes

Não existe hoje nenhum arquivo de teste no repositório (`find ... -iname "test_*.py"` =
vazio). Proposta em três camadas, para dar cobertura real sem depender só de testes
manuais na UI:

### 5.1 Testes unitários (sem Isaac Sim, rápidos, rodam em CI comum)

| Teste | O que guarda contra regressão |
|---|---|
| Falloff de intensidade do sonar vs. distância (função de referência em NumPy espelhando o kernel Warp) | Reintrodução do bug de normalização por anel que achata a distância |
| Round-trip `world_to_voxel`/`voxel_to_world` e `freespace_mask` em grids sintéticos | Erros de indexação/origem no novo mapa de ocupação (2D e 3D) |
| `Config.to_json`/`from_json` com todos os campos novos | Quebra de compatibilidade de config entre versões |
| Round-trip `Writer`→`Reader` de um `state_dict` sintético (todas as modalidades) | Regressão de serialização (pickling, PNG16, JSON de metadados) |
| Função de compensação de escala por `metersPerUnit` | Reintrodução do bug de escala ao inserir assets |
| Mapeamento de eixos do gamepad → força/torque 6DOF (ganho, deadzone, expo) | Comportamento errado de pilotagem, incluindo sinal invertido |

### 5.2 Testes de integração (Isaac Sim headless, mais lentos)

| Teste | O que guarda contra regressão |
|---|---|
| Build de cada cenário registrado × robô ROV por N passos de física, sem exceção | Quebra de qualquer cenário após mudanças na física/sensores |
| Gravação ao vivo curta com todos os sensores ligados (câmera, estéreo, sonar, lidar, DVL, barômetro): todo frame tem o conjunto completo de chaves esperado em `state/common` | Falhas silenciosas de sensor voltando a acontecer (guarda contra reintrodução do `except: pass`) |
| Arquivos brutos de sonar (`.npy`/PNG16/polar) realmente gravados durante gravação **ao vivo** (não só replay) | Regressão da Fase 0.3 |
| Nuvem de pontos do lidar não-vazia e dentro do alcance máximo configurado | Perfil de lidar subaquático corretamente aplicado |
| Geração do mapa de ocupação em 2–3 mundos de teste: proporção não-trivial de células livres/ocupadas | Regressão para o fallback silencioso "tudo livre" |
| Para o modo 3D: contagem de células ocupadas varia de forma coerente entre bandas de profundidade | Bandas degeneradas/idênticas no gerador 3D |
| Validação de material/textura de cada asset em `assets/models/*`: nenhum caminho de textura não resolvido | Regressão do bug de asset sem textura |
| Bounding box de cada asset dentro de uma faixa de tamanho plausível por categoria | Regressão do bug de escala |
| Pose: comandar surge puro e checar que o deslocamento é predominantemente no eixo frontal do robô; pose de cada sensor mantém offset constante em relação à pose do robô em todos os frames | Valida a suposição de montagem rígida e detecta dessincronia sensor/robô |
| Cenário de gamepad com valores de eixo sintéticos (sem hardware físico): saída de força/torque esperada; desconexão no meio zera a ação | Regressão de segurança na teleoperação |
| `Waypoint`/`PathFollowing`: robô alcança o alvo dentro de tolerância e nunca entra em célula ocupada | Garantia de navegação livre de colisão |

### 5.3 Testes de qualidade de dado (ligados ao uso final em GS/fusão sensorial)

| Teste | Motivo |
|---|---|
| Alvo refletivo posicionado a distâncias conhecidas numa cena de teste: perfil de intensidade do sonar segue o decaimento exponencial esperado dentro de tolerância | Confirma ponta-a-ponta que a correção do sonar realmente restaura sinal de distância utilizável — crítico pois `DPS_sonar_net`/`PSNetSonarPriority` consomem esse sinal como parte do treino |
| Cobertura angular mínima de views numa cena de varredura de objeto (câmera orbitando um alvo) | Garante que o dataset gerado é utilizável para reconstrução 3DGS — poucas views/baixa cobertura angular já quebrou reconstruções COLMAP em outros datasets do projeto (ver histórico do DAVE scene1 no `planejamento_pesquisa`) |

---

## 6. Itens em aberto (para decidir antes/durante a implementação)

1. **Implementação de `water_profile_path`** ainda não foi inspecionada a fundo — precisa
   ser lida antes da Fase 2 para decidir se vira a fonte única de turbidez para sonar e
   lidar, ou se cada sensor mantém seu próprio parâmetro.
2. **Suporte a botões do gamepad**: no MobilityGen de referência, leitura de botão está
   stub (`get_button_values()` retorna zero). Se quiser alternar "modo preciso/turtle"
   por botão em vez de por trigger analógico, isso precisa ser implementado antes.
3. **Planejador de caminho 3D (Fase 4b)**: decidir se vale a pena portar/estender o
   `path_planner` C++ existente (2D) para 3D, ou começar com A* em Python puro (mais
   simples, mais lento) e otimizar depois se necessário.
4. **Alcance real do sonar simulado vs. faixa operacional do dataset**: vale confirmar
   com o usuário se o `attenuation=0.3` atual do sonar está calibrado para a faixa de
   distância que os datasets reais (DAVE, Eilat, etc.) cobrem, para escolher o preset de
   normalização default de forma consistente com o domínio real.

---

## 7. Próximos passos

Este documento cobre **apenas avaliação e planejamento**, conforme solicitado. Após
revisão, a implementação pode ser conduzida fase a fase (Fase 0 → Fase 5), cada uma
encerrando com os testes da seção 5 correspondentes antes de avançar para a próxima.
