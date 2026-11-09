# Relatório de implementação — v2: navegação leve + World Builder

Continuação de
[`relatorio_implementacao_upgrade_underwater.md`](relatorio_implementacao_upgrade_underwater.md)
(rodada 1: correções de sonar/assets/pose/lidar/joystick/occupancy map). Esta
rodada implementa dois pedidos novos: (1) deixar o modo de navegação mais
leve/fluido desligando sensores por padrão, e (2) um launch separado para
criação/validação de mundos, sem robô, com ferramentas de física e
reflectância do sonar. Commitado em 09/11/2026 (`efb77b2`), junto com a integração descrita na
seção 5.

---

## 1. Navegação leve por padrão

**Problema**: ao abrir o `omni.ext.patosim` e construir um cenário, todos os
sensores configurados (câmera, estéreo, sonar, lidar, DVL, barômetro) já
ficavam ativos mesmo que o objetivo fosse só pilotar o ROV pela cena
(navegação/teste), pesando a simulação sem necessidade.

**Correção**: os defaults de todos os 6 flags `enable_rov_*` foram trocados
para `False`, em dois lugares que precisavam ficar consistentes:

- `exts/omni.ext.patosim/omni/ext/patosim/config.py` (`Config` dataclass) —
  usado por qualquer construção programática/headless de `Config`.
- `exts/omni.ext.patosim/omni/ext/patosim/extension.py` (`_oceansim_front_camera_model`,
  `_oceansim_stereo_camera_model`, `_oceansim_lidar_model`, `_oceansim_sonar_model`,
  `_oceansim_dvl_model`, `_oceansim_barometer_model`) — os checkboxes reais da UI,
  que é o que `create_config()` de fato lê quando você clica "Build".

As checkboxes continuam lá, funcionando exatamente como antes — você liga o
que precisar antes de gravar. Nada na lógica de navegação (teleop por
teclado, joystick, waypoint, path-following) depende de DVL/barômetro para
controle, então desligá-los por padrão não quebra a pilotagem (confirmado
na avaliação da rodada 1: esses sensores só alimentam o dataset gravado,
não o laço de controle).

Nota: isso reverte o default `enable_rov_lidar=True` que eu tinha definido
na rodada 1 (Fase 2) — ali o pedido era "ligar o lidar por padrão" para o
caso de uso de gravação; agora o pedido explícito é o oposto para o caso de
uso de navegação. O modelo de atenuação subaquática do lidar (seção 4 do
relatório v1) continua intacto, só o default de ligado/desligado muda.

---

## 2. Novo launch: World Builder (sem robô)

### Por que uma extensão separada, e não um modo dentro da mesma

Optei por criar `exts/omni.ext.patosim_worldbuilder` como uma **segunda
extensão Kit**, em vez de um modo/aba dentro do `omni.ext.patosim` existente,
pelas seguintes razões:

1. Você pediu explicitamente "um outro launch" — não uma aba na mesma janela.
2. "Use como base apenas as ferramentas necessárias que foram utilizadas do
   oceansim" — uma extensão separada deixa isso literal: ela não carrega (e
   não corre o risco de acidentalmente disparar) nenhum código de
   robô/sensor/gravação do `omni.ext.patosim`.
3. O próprio `PATOSIM_LAYOUT.md` já documentava a intenção de extensões
   modulares desde o início do projeto.

A nova extensão **reaproveita** (não duplica) a lógica pura de
`omni.ext.patosim.build` — ela importa esse módulo diretamente via Python
(resolvendo o caminho do pacote-irmão manualmente em `extension.py`, sem
declarar `omni.ext.patosim` como dependência Kit), então não existe duas
cópias de "como aplicar colisão"/"como aplicar reflectância" para
divergirem no futuro (o mesmo tipo de bug que causou o problema do §0 do
relatório v1).

### O que a nova extensão faz

Arquivo: `exts/omni.ext.patosim_worldbuilder/omni/ext/patosim_worldbuilder/extension.py`.
Lançada via `scripts/launch_world_builder.sh` (`--enable
omni.ext.patosim_worldbuilder`, **sem** habilitar `omni.ext.patosim`).

- **Load World (no robot)** — `build_world_only(scene_usd)` (nova função
  pública em `build.py`): cria stage novo, referencia a cena em
  `/World/scene`. Nenhum robô é instanciado em lugar nenhum desse fluxo.
- **Import Dataset Object** — reaproveita `list_dataset_object_assets()` e
  `_insert_dataset_object_into_scene()` (já existentes, usados pelo picker
  de objetos do extension principal) para listar e inserir um asset de
  `assets/models/{plataforms,statues_temples,scenario}`, com campos de
  posição/rotação/escala/reflectividade.
- **Check Semantic Classes** — nova função pública `check_scene_semantics()`
  em `build.py`: percorre todos os `UsdGeom.Mesh` da cena e reporta quantos
  já têm os labels semânticos `class` e `reflectivity` (os que o
  `compute_intensity()` do sonar usa para buscar a reflectividade por id
  semântico — sem esse label, o mesh usa um valor default, que pode parecer
  visualmente ok mas está errado para dataset de sonar). **Não modifica
  nada** — é só leitura, usando `isaacsim.core.utils.semantics.get_semantics()`
  (confirmei a assinatura real lendo o código-fonte instalado do Isaac Sim
  em `app/exts/isaacsim.core.utils/isaacsim/core/utils/semantics.py`, não
  advinhei).
- **Apply Default Reflectivity** — chama a função que já existia
  (`_apply_sonar_reflectivity_to_scene`, usada no fluxo normal de build) com
  o valor default que você define no campo.
- **Check Physics** — nova função pública `check_scene_physics()`: percorre
  todos os `UsdGeom.Gprim` e reporta quantos já têm `UsdPhysics.PhysxCollisionAPI`
  aplicado. Usei exatamente o mesmo schema que `isaacsim.core.prims.SingleGeometryPrim`
  checa internamente (confirmei lendo `app/exts/isaacsim.core.prims/isaacsim/core/prims/impl/geometry_prim.py`),
  para que "check" e "apply" concordem sobre o que conta como "tem colisão".
- **Apply Physics (Collisions)** — chama `_enable_scene_collisions` (já
  existente).

O fluxo pretendido: abrir o World Builder, carregar o mundo, rodar os dois
"Check", corrigir o que faltar com os dois "Apply" (ou ajustar o asset/USD
na origem se o problema for mais sério, ex.: escala), e só então abrir o
`omni.ext.patosim` normal para montar o cenário de gravação na mesma cena.

### Arquivos novos desta rodada

```
exts/omni.ext.patosim_worldbuilder/
    config/extension.toml
    docs/{README.md,CHANGELOG.md}
    omni/ext/patosim_worldbuilder/{__init__.py,extension.py}
scripts/launch_world_builder.sh
```

E em `exts/omni.ext.patosim/omni/ext/patosim/build.py`: 3 funções públicas
novas (`build_world_only`, `check_scene_semantics`, `check_scene_physics`) —
nenhuma função existente foi removida ou teve seu comportamento alterado.

---

## 3. Testes pendentes agora executados com GPU real (2026-10-24)

A seção anterior (preservada no histórico do git) listava um checklist
manual porque esta sessão não tinha GPU/Kit disponível. Com a GPU liberada,
rodei tudo de verdade contra o Isaac Sim real desta máquina (RTX 4070,
driver 565.77) — nenhum resultado abaixo foi inventado ou extrapolado.

### Como foi testado

Dois scripts novos em `tests/integration/`, executados com
`./app/python.sh tests/integration/<script>.py` (não precisam da UI — rodam
headless, via `isaacsim.SimulationApp`, e chamam as mesmas funções de
produção que os botões da UI chamam):

- **`run_gpu_integration_checks.py`**: builda cenários ROV completos de
  verdade (`build_scenario_from_config`), roda física por múltiplos passos,
  grava uma sessão curta e inspeciona os arquivos gerados, gera o stack 3D
  de occupancy map, testa o controller de gamepad, e testa retrocompatibi-
  lidade carregando o `config.json` de uma gravação real pré-existente
  (`~/PatoSimData/recordings/2026-03-24T17:29:01.269957`, que antecede todos
  os campos novos de `Config` desta sessão).
- **`run_worldbuilder_extension_check.py`**: habilita a extensão World
  Builder pelo gerenciador de extensões real do Kit e confirma que a
  extensão principal (`omni.ext.patosim`) **não** é carregada junto.

Importante sobre fidelidade: as duas primeiras tentativas falharam porque
`SimulationApp` sozinho só carrega um conjunto mínimo de extensões — não o
conjunto completo que `launch_patosim.sh` carrega. Corrigi isso habilitando
pelo gerenciador de extensões real do Kit (`isaacsim.asset.gen.omap` e o
provedor do namespace `isaacsim.oceansim`) em vez de um atalho de
`sys.path`, o que tornou o teste mais fiel ao que um usuário real roda — e,
como efeito colateral, confirmou de verdade a consolidação por symlink da
seção 0 do relatório v1 (a correção mais arriscada daquela rodada): o
provedor do namespace `isaacsim.oceansim` descoberto e habilitado foi
`PatoSIM-1.0.0`, que aponta (via symlink) para o código corrigido.

### Bugs reais encontrados e corrigidos durante esta sessão de testes

| Bug | Onde | Causa | Correção |
|---|---|---|---|
| `check_scene_physics` sempre reportava 0 colisões, mesmo depois de aplicar | `build.py` (função nova da rodada 2) | Eu checava `UsdPhysics.PhysxCollisionAPI`, mas `isaacsim.core.prims.GeometryPrim.apply_collision_apis()` aplica `UsdPhysics.CollisionAPI` (a API base), não essa variante PhysX. Confirmei lendo `apply_collision_apis()` no código-fonte real do Isaac Sim instalado. | Troquei o schema checado para `UsdPhysics.CollisionAPI`. Depois da correção: antes do apply, 234/428 geoms já tinham colisão (vinda do próprio asset); depois do apply, 428/428 — números corretos e coerentes pela primeira vez. |
| `omni.ext.patosim_worldbuilder/extension.py` nunca conseguiria importar `omni.ext.patosim.build` de verdade | extensão nova da rodada 2 | `omni.ext` é um pacote real do Kit, não um namespace package — um `sys.path.insert` sozinho não basta para `import omni.ext.patosim` resolver (confirmado reproduzindo o erro exato: `ModuleNotFoundError: No module named 'omni.ext.patosim'`). Faltava o segundo passo que `scripts/replay_implementation.py` já fazia: `omni.ext.__path__.append(...)`. | Adicionei esse segundo passo ao bootstrap do `extension.py` da nova extensão. Confirmado funcionando: a janela "PatoSim - World Builder" abre de verdade pelo gerenciador de extensões do Kit. |
| (no meu próprio script de teste, não em código de produção) mesmo erro de bootstrap, e esquecimento de habilitar o provedor do `isaacsim.oceansim` | `tests/integration/run_gpu_integration_checks.py` | — | Corrigido iterativamente nas primeiras tentativas; documentado aqui só por transparência do processo. |

### Bug real encontrado e **não corrigido** (decisão sua)

**`_compute_safe_underwater_spawn()` (`build.py`) deixa o ROV flutuando
acima de todo o mundo, não dentro da água, para qualquer cena configurada
de forma sensata.**

- A função assume que, se o ponto de spawn configurado está "dentro" do
  bounding box da cena, é arriscado (risco de nascer dentro de um
  colisor/geometria sólida) e levanta o robô para `max_z_da_cena + 1.0`.
- Mas para uma cena subaquática, "dentro do bounding box da cena" é
  exatamente a condição **normal e esperada** — é onde fica o volume de
  água! Confirmei isso rodando de verdade: com a cena padrão
  (`Prototipo1/world.usd`, bbox de z=-6.56 a z=5.73), o ROV nasceu em
  z=6.73 — ou seja, 1 metro **acima do ponto mais alto de toda a cena**,
  flutuando no ar/vácuo acima do mundo, não submerso.
- Isso explica por que, inicialmente, sonar e lidar não detectavam nada
  (não havia nada a até 10m de distância ali em cima) — não era um bug no
  sonar, era o robô nascendo no lugar errado.
- **Não corrigi isso agora** porque a correção certa depende de uma decisão
  de design (ex.: trocar "está dentro do bounding box" por um raycast para
  baixo a partir do ponto desejado, checando se há geometria sólida
  literalmente ali, não apenas "dentro da caixa que envolve toda a cena") —
  e o pedido desta rodada era rodar testes, não implementar correções
  novas. Fica registrado aqui como achado prioritário para a próxima rodada.

### Resultado consolidado (18 de ~20 checks diretamente relevantes passaram)

| Check | Resultado | Detalhe |
|---|---|---|
| Config retrocompatível com gravação antiga | ✅ | `config.json` de 2026-03-24 (antes de todos os campos novos) carrega e usa os defaults novos corretamente |
| World Builder: check antes do apply | ✅ | 428 meshes, 0 com classe/reflectividade, 234/428 já com colisão (vinda do asset) |
| World Builder: apply reflectância | ✅ | 0 → 428 meshes com reflectividade |
| World Builder: apply física | ✅ | 234 → 428 geoms com colisão (depois da correção do bug acima) |
| World Builder: listar/inserir dataset object | ✅ | 11 assets encontrados; inserção confirmada com prim válido no stage |
| **World Builder: extensão carrega sem a extensão principal** | ✅ | Habilitada pelo gerenciador real do Kit; `omni.ext.patosim` confirmado **não** habilitado junto; janela "PatoSim - World Builder" criada de verdade |
| Build de cenário ROV completo (sonar+lidar+dvl+barômetro+câmera) | ✅ | Todos os 5 sensores construídos sem erro |
| `sonar_normalizing_method="raw"` chega até o sensor de verdade | ✅ | `robot.sonar._normalizing_method == "raw"` confirmado no objeto real |
| Perfil de lidar `insight_micro` chega até o sensor de verdade | ✅ | `robot.lidar` com min/max = 0.13/7.0 confirmado |
| Gravação ao vivo: `sensors_extrinsics.json` | ✅ | Arquivo real escrito e não-vazio |
| Gravação ao vivo: `state/common/*.npy` com chaves de pose | ✅ | 15 arquivos, chaves `robot.*`/sensor.* presentes |
| **Gravação ao vivo: sonar bruto (`state/sonar/raw/*.npy`)** | ✅ | 15 arquivos escritos **durante a gravação ao vivo** — este é exatamente o fix da Fase 0.3 do relatório v1, agora confirmado funcionando de ponta a ponta, não só em teoria |
| Pasta `occupancy_map_3d/` com 3 bandas | ✅ | `manifest.json` real gerado, 3 bandas com z_centers corretos |
| Gamepad: conecta headless sem hardware físico, ação zerada | ✅ | Confirma o design de segurança da Fase 3 do relatório v1 |
| Sonar/lidar detectando algo de verdade a uma distância conhecida | ⚠️ inconclusivo | Ver abaixo |

### O item inconclusivo: sinal real não-degenerado do sonar/lidar

Tentei 4 vezes, de formas diferentes, fazer o sonar/lidar "verem" algo de
verdade nesta cena de teste (`Prototipo1/world.usd`):

1. Pose de spawn padrão → nada em alcance (explicado pelo bug acima).
2. Robô teleportado para o centro da cena, nivelado → nada em alcance.
3. Robô teleportado + apontado para baixo (pitch -90°) → o sonar chegou a
   dar erro interno nessa pose extrema (capturado, não travou o processo).
4. Robô reposicionado para dentro da água (corrigindo manualmente o bug de
   spawn, só neste teste) + um objeto conhecido inserido 3m na frente dele
   → ainda nada detectado.

Em todas as tentativas, o sonar reportou status `"sonar_frame"` (ou seja,
rodou sem lançar exceção) e devolveu um array com a forma certa, só que
com todos os valores zero; o lidar reportou `"rtx_no_pointcloud"` (zero
retornos, não um erro). Isso **não contradiz a correção do modo "raw"** —
o teste unitário puro (`tests/unit/test_sonar_normalization_math.py`) já
prova matematicamente, de forma rigorosa, que o algoritmo preserva o
decaimento por distância; o que ficou sem confirmação aqui é especificamente
"o sonar detecta algo de verdade nesta cena, nesta pose". As causas mais
prováveis que não tive tempo de isolar: a orientação de montagem do sonar/
lidar (`sonar_orientation_euler_deg=(0,0,0)`) pode não apontar para onde eu
assumia (frente do corpo = +X mundo) sem mais investigação do FOV da
câmera do sonar; ou o objeto de teste inserido pode ter algum problema de
escala/geometria específico. **Recomendação prática**: abrir
`./scripts/launch_patosim.sh` de verdade, pilotar o ROV manualmente até
perto de uma parede/estrutura visível na viewport, e abrir a janela de
preview do sonar — isso resolve a ambiguidade em segundos, de um jeito que
eu não consigo replicar às cegas via terminal.

### Suite de testes puros (`tests/unit/`)

Sem mudanças de resultado: continuam 32/32 passando depois de todas as
correções desta sessão.

## 4. Próximos passos

Mesmos da rodada 1 (seção 4 do relatório anterior), mais:
7. **Prioridade alta**: corrigir `_compute_safe_underwater_spawn()` (seção 3
   acima) — o ROV nasce fora da água em qualquer cena configurada de forma
   normal.
8. Confirmar visualmente (UI real) que o sonar/lidar detectam estruturas
   conhecidas da cena, fechando o item inconclusivo da seção 3.
9. Se quiser, adicionar um botão "Build Recording Scenario Here" no World
   Builder que simplesmente abre/foca a janela do `omni.ext.patosim` com o
   mesmo `scene_usd` já preenchido, para não precisar reabrir o caminho à mão.

---

## 5. Integração no `main`, instalação na pdi_mini_3 e revisão dos testes (09/11/2026)

### O que entrou no `main` (e no `master`, igual)

- `efb77b2`: esta rodada (navegação leve, World Builder, testes de integração).
- Merge do `origin/main` (`f3a4728`, correção do caminho do ROV e `.gitignore`). O `.gitignore` passou a
  ignorar só os artefatos de build do `path_planner`; os fontes dele continuam versionados.
- Merge da branch `update_imports` (feita na pdi_mini_3): o código do OceanSim é importado como
  `omni.ext.patosim.oceansim` em todo lugar. Antes, `isaacsim.oceansim` vinha de uma extensão em
  `app/extsUser`, fora do git (a seção 0 do relatório v1). Na pdi_mini_3 essa cópia era de março, então
  o sonar corrigido não rodaria lá. O teste de integração agora confere de onde o `ImagingSonarSensor`
  foi carregado.
- Caminhos de uma máquina só, trocados por caminhos relativos ao repositório: ROV e objetos do dataset em
  `assets/models/` ou `assets/` (layout da pdi_mini_3); mundo do botão "Use MHL Scene" (ou
  `$PATOSIM_DEV_SCENE`); link `isaacsim/oceansim` (era absoluto para `/home/pdi_4/...`); scripts de
  `tests/integration` (gravação antiga opcional, `$PATOSIM_LEGACY_RECORDING`).
- `8c92c04`: no teste, o ROV fica 1,5 m acima da primeira superfície achada por um raycast para baixo.
  Antes ele ia para "mínimo da caixa da cena + 3 m", e no Prototipo1 isso fica **abaixo do fundo do
  mar**: o fundo está em z≈0 m e os corais descem até z=-4,9 m.

### Instalação na pdi_mini_3

`~/Documents/PatoSIM` passou da branch `update_imports` para o `main`. Os assets fora do git, em `assets/`,
não foram tocados. Depois o procedimento do README: `pip install tqdm wheel`, `pybind11` e
`path_planner` com `--no-build-isolation -e .`, que recompilou o módulo C++. O `.venv` do repo foi criado
quando a pasta se chamava `~/Documentos/PatoSIM` e não funciona mais; os testes unitários usam o
`python3` do sistema.

### Resultados

| Checagem | Esta máquina (RTX 4070) | pdi_mini_3 (RTX 5060 Ti) |
|---|---|---|
| `tests/unit` | 32/32 | 32/32 |
| `run_gpu_integration_checks.py` | 25 ok, 1 falha (sinal do sonar), crash na última checagem | 26 ok (2 pulados), 1 falha (sinal do sonar) |
| `run_worldbuilder_extension_check.py` | 6/6 | 6/6 |

- **Sinal do sonar**: continua zerado no modo headless nas duas máquinas, agora com o ROV acima do fundo
  e um objeto 3 m à frente. A posição não era a causa, ou não era a única. Pela interface o sonar gera
  imagem (gravação de 24/03). O item da seção 3 continua aberto, restrito ao modo headless.
- **Spawn (seção 3)**: o robô não fica "no ar". O OceanSim não tem volume de água, a água é um efeito de
  renderização. O que acontece é que ele nasce 1 m acima do objeto mais alto do mundo (z=6,7 m no
  Prototipo1). Com o fundo em z≈0 m, a profundidade padrão de -2 m fica abaixo do fundo.
- **Gravação antiga**: nesta máquina, carregar a gravação de 24/03 no mesmo processo derruba o Isaac
  (crash nativo em `omni.usd` ao criar o `World` de novo, dentro de `load_scenario`). As outras checagens
  já estão salvas nesse ponto. Na pdi_mini_3 a checagem é pulada.

Documentação: README reescrito para o fluxo subaquático e `docs/PatoSim_novidades.pdf` (fonte em
`docs/manual/`), com as novidades e o guia de uso.
