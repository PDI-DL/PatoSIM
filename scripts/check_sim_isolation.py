#!/usr/bin/env python3
"""
Checagem de isolamento entre os simuladores que nasceram do MobilityGen (BEVSIM e PatoSim) numa máquina.

Os dois são plugins diferentes do Isaac Sim com a mesma origem: dividem nomes (extensão omni.ext.mobility_gen no
BEVSIM e no legacy_exts do PatoSim, pacote mobility_gen_path_planner, env conda "mobgen"), e às vezes o mesmo Isaac.
Este script mede, para o repositório escolhido, cada ponto em que um pode pegar algo do outro ou de outra máquina:

  repo        o repositório é do simulador certo (remote e pasta da extensão)
  isaac       app/ aponta para um Isaac que existe; quem mais usa o mesmo Isaac
  extsUser    extensões que o Isaac registra em TODA sessão (inclusive as deste plugin)
  kit_config  extensões e pastas salvas na configuração do Kit (vale para todos os Isaac 5.1 do usuário)
  planner     de onde o Python do Isaac importa o mobility_gen_path_planner
  conda       envs com o mesmo nome em environment.yml de simuladores diferentes
  shell       variáveis e atalhos de shell que valem para os dois simuladores
  atalhos     entradas de menu que abrem o Isaac (com ou sem este plugin)
  assets      assets do robô/sensores que não existem nesta máquina (caminhos fixos de outra)
  caminhos    caminhos absolutos de outra máquina no código
  mundos      referências dos mundos que não resolvem (precisa do pxr: rode com o env mobgen)
  sessoes     o que as últimas sessões do Isaac deste repositório carregaram de verdade (logs do Kit)
  processos   Isaac rodando, inclusive órfãos

    python3 scripts/check_sim_isolation.py                 # este repositório
    python3 scripts/check_sim_isolation.py --repo ~/PatoSIM
    python3 scripts/check_sim_isolation.py --json saida.json

Saída: uma linha por achado ([ERRO]/[AVISO]/[OK]) e o resumo; o código de saída é o número de ERROs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

HOME = Path.home()
KINDS = {"omni.ext.mobility_gen": "BEVSIM", "omni.ext.patosim": "PatoSim"}
REMOTES = {"BEVSIM": "Automni-BEVLOG/BEVSIM", "PatoSim": "PDI-DL/PatoSIM"}
FOREIGN_WORDS = {"BEVSIM": ("oceansim", "patosim", "ocean"), "PatoSim": ("bevsim", "forklift")}
SEARCH_ROOTS = ["~", "~/Documentos/BEVLOG/SIMULADOR", "~/BEVLOG", "~/Documents", "~/Documents/Documentos/BEVLOG/bevlog-isaac",
                "~/Documentos", "/mnt/external/isaac"]

findings: list[dict] = []


def add(level: str, cat: str, msg: str, **extra) -> None:
    findings.append({"nivel": level, "categoria": cat, "msg": msg, **extra})


def real(p) -> str:
    return os.path.realpath(os.path.expanduser(str(p)))


def kind_of(repo: Path) -> str:
    for ext, kind in KINDS.items():
        if (repo / "exts" / ext).is_dir():
            return kind
    return ""


def toml_modules(path: Path) -> list[str]:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    return re.findall(r'\[\[python\.module\]\]\s*\n\s*name\s*=\s*"([^"]+)"', text)


def git(repo: Path, *args) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def find_sim_repos(extra: list[str]) -> list[Path]:
    """Repositórios de simulador (pasta exts/ com a extensão de um deles) até 2 níveis abaixo das raízes."""
    found = set()
    for root in SEARCH_ROOTS + extra:
        base = Path(os.path.expanduser(root))
        if not base.is_dir():
            continue
        for depth_glob in ("*", "*/*"):
            for cand in base.glob(depth_glob):
                try:
                    if cand.is_dir() and not cand.name.startswith(".") and kind_of(cand):
                        found.add(real(cand))
                except OSError:
                    continue
    return sorted(Path(p) for p in found)


# ----------------------------------------------------------------------------------------------------------------------
def check_repo(repo: Path, kind: str) -> None:
    url = git(repo, "remote", "get-url", "origin")
    if kind == "BEVSIM" and (repo / "exts" / "omni.ext.patosim").is_dir():
        add("ERRO", "repo", "o repositório do BEVSIM tem exts/omni.ext.patosim (pasta do PatoSim)")
    if REMOTES[kind] not in url:
        add("ERRO", "repo", f"remote '{url}' não é o do {kind} ({REMOTES[kind]})")
    else:
        add("OK", "repo", f"{kind}: {repo} · {git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')} "
                          f"{git(repo, 'log', '--oneline', '-1')[:60]} · {url}")


def check_isaac(repo: Path, repos: list[Path]) -> Path | None:
    app = repo / "app"
    if not app.exists():
        add("ERRO", "isaac", f"app/ não existe ou aponta para um caminho que não existe ({os.readlink(app) if app.is_symlink() else 'sem link'})")
        return None
    isaac = Path(real(app))
    version = (isaac / "VERSION").read_text().split("+")[0].strip() if (isaac / "VERSION").is_file() else "?"
    add("OK", "isaac", f"app/ -> {isaac} (Isaac {version})")
    sharing = [r for r in repos if real(r) != real(repo) and (r / "app").exists() and real(r / "app") == str(isaac)]
    for r in sharing:
        add("AVISO", "isaac", f"o mesmo Isaac é usado por {kind_of(r)} em {r}: extsUser, pacotes do Python do Isaac e "
                              "configuração do Kit são compartilhados")
    for r in repos:
        if (r / "app").is_symlink() and not (r / "app").exists():
            add("AVISO", "isaac", f"{kind_of(r)} em {r}: app/ quebrado -> {os.readlink(r / 'app')}")
    return isaac


def check_extsuser(isaac: Path, kind: str, own_modules: list[str]) -> None:
    folder = isaac / "extsUser"
    entries = sorted(p for p in folder.iterdir() if (p / "config" / "extension.toml").is_file()) if folder.is_dir() else []
    if not entries:
        add("OK", "extsUser", f"{folder} sem extensões (nada registrado em toda sessão)")
        return
    by_module: dict[str, list[str]] = {}
    for ext in entries:
        mods = toml_modules(ext / "config" / "extension.toml")
        for m in mods:
            by_module.setdefault(m, []).append(ext.name)
        clash = [m for m in mods if m in own_modules or m.startswith(tuple(o + "." for o in own_modules))]
        if clash:
            add("ERRO", "extsUser", f"{ext} fornece {clash}, o mesmo módulo deste plugin: pode carregar no lugar dele")
        elif any(w in (ext.name + " ".join(mods)).lower() for w in FOREIGN_WORDS[kind]):
            add("AVISO", "extsUser", f"{ext.name} ({', '.join(mods) or 'sem módulo'}) é de outro simulador e é registrada "
                                     f"em toda sessão deste Isaac")
        else:
            add("AVISO", "extsUser", f"{ext.name} ({', '.join(mods) or 'sem módulo'}) registrada em toda sessão")
    for mod, exts in by_module.items():
        if len(exts) > 1:
            add("AVISO", "extsUser", f"o módulo {mod} está em {len(exts)} pastas ({', '.join(exts)}): qual carrega depende "
                                     "da ordem do Kit")


def check_kit_config(kind: str, own_ext: str) -> None:
    files = glob.glob(str(HOME / ".local/share/ov/data/Kit/*/*/user.config.json"))
    hits = 0
    for f in files:
        try:
            data = json.load(open(f))
        except Exception:
            continue
        def walk(o, p=""):
            nonlocal hits
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{p}/{k}")
            elif isinstance(o, list):
                for v in o:
                    walk(v, p)
            elif isinstance(o, str):
                low = o.lower()
                if ("exts" in p.lower() and ("enabled" in p.lower() or "folders" in p.lower())) or \
                        any(w in low for w in ("oceansim", "patosim", own_ext)):
                    if "/exts/" in p and not any(w in low for w in ("oceansim", "patosim", "mobility_gen")):
                        return
                    hits += 1
                    level = "ERRO" if any(w in low for w in FOREIGN_WORDS[kind]) and "enabled" in p.lower() else "AVISO"
                    add(level, "kit_config", f"{Path(f).parent.parent.name}: {p} = {o[:160]}")
        walk(data)
    if not hits:
        add("OK", "kit_config", f"{len(files)} configurações do Kit sem extensão ou pasta de simulador salva")


def check_planner(isaac: Path, repo: Path) -> None:
    site = next(iter(glob.glob(str(isaac / "kit/python/lib/python3.*/site-packages"))), "")
    infos = glob.glob(os.path.join(site, "mobility_gen_path_planner-*.dist-info")) if site else []
    own = repo / "path_planner"
    if not infos:
        add("OK" if own.is_dir() else "AVISO", "planner", "mobility_gen_path_planner não instalado no Python do Isaac: "
            "cada extensão usa o path_planner do próprio repositório (sys.path)")
    for info in infos:
        target = ""
        try:
            target = json.load(open(os.path.join(info, "direct_url.json"))).get("url", "").replace("file://", "")
        except Exception:
            pass
        if not target:
            add("AVISO", "planner", f"{Path(info).name} instalado sem caminho de origem (cópia, não editável)")
        elif not os.path.isdir(target):
            add("ERRO", "planner", f"o path_planner instalado no Isaac aponta para {target}, que não existe")
        elif real(target) != real(own):
            add("ERRO", "planner", f"o path_planner instalado no Isaac é o de {target}, não o deste repositório ({own})")
        else:
            add("OK", "planner", f"path_planner instalado no Isaac = {target}")
    sos = glob.glob(str(own / "mobility_gen_path_planner" / "_mobility_gen_path_planner_C*.so"))
    if own.is_dir() and not sos:
        add("AVISO", "planner", "path_planner sem a extensão C++ compilada: usa a versão em Python (lenta); rode "
                                "./app/python.sh -m pip install --no-build-isolation -e path_planner")


def check_conda(repos: list[Path]) -> None:
    names: dict[str, list[str]] = {}
    for r in repos:
        f = r / "environment.yml"
        if f.is_file():
            m = re.search(r"^name:\s*(\S+)", f.read_text(errors="ignore"), re.M)
            if m:
                names.setdefault(m.group(1), []).append(f"{kind_of(r)} ({r})")
    for name, who in names.items():
        kinds = {w.split(" ")[0] for w in who}
        if len(kinds) > 1:
            add("ERRO", "conda", f"o env '{name}' é definido por simuladores diferentes: {'; '.join(who)}. Criar/atualizar "
                                 "o de um apaga as versões do outro")
        else:
            add("OK", "conda", f"env '{name}' só em {', '.join(who)}")


def check_shell(kind: str) -> None:
    rc_files = [HOME / f for f in (".bashrc", ".profile", ".bash_aliases", ".zshrc")]
    pat = re.compile(r"(MOBILITY_GEN_DATA|PATOSIM_DATA|ISAAC_SIM_PATH|isaac-sim\.sh|launch_sim\.sh|launch_patosim|"
                     r"bevsim_launcher|patosim\b)")
    for f in rc_files:
        if not f.is_file():
            continue
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if line.strip().startswith("#") or not pat.search(line):
                continue
            level = "AVISO"
            if re.search(r"export\s+MOBILITY_GEN_DATA", line):
                level = "ERRO" if "pato" in line.lower() else "AVISO"
            add(level, "shell", f"{f.name}:{i}: {line.strip()[:150]}")
    for var in ("MOBILITY_GEN_DATA", "PATOSIM_DATA", "ISAAC_SIM_PATH"):
        if os.environ.get(var):
            add("AVISO", "shell", f"{var}={os.environ[var]} no ambiente atual")


def check_shortcuts(repo: Path, own_ext: str) -> None:
    dirs = [HOME / ".local/share/applications", HOME / "Desktop", HOME / "Área de Trabalho"]
    for d in dirs:
        for f in sorted(d.glob("*.desktop")) if d.is_dir() else []:
            text = f.read_text(errors="ignore")
            m = re.search(r"^Exec=(.*)$", text, re.M)
            if not m or not re.search(r"isaac-sim\.sh|kit/kit|launch_sim|launch_patosim|bevsim_launcher|desktop\.py", m.group(1)):
                continue
            exe = m.group(1).split()[0]
            if not os.path.exists(exe):
                add("ERRO", "atalhos", f"{f.name}: Exec aponta para {exe}, que não existe")
            elif own_ext in m.group(1) or "bevsim_launcher" in m.group(1) or "launch_sim" in m.group(1):
                add("OK", "atalhos", f"{f.name}: abre o simulador com o plugin ({m.group(1)[:100]})")
            else:
                add("AVISO", "atalhos", f"{f.name}: abre o Isaac puro, SEM o plugin deste repositório ({exe}); a janela "
                                        "MobilityGen que aparecer ali é a do Isaac (isaacsim.replicator.mobility_gen)")


def check_assets(repo: Path) -> None:
    """Cada _resolve_asset_path(...) precisa de pelo menos um candidato que exista nesta máquina."""
    bad = 0
    sources = {py: py.read_text(errors="ignore") for py in sorted((repo / "exts").glob("*/omni/ext/*/*.py"))}
    everything = "\n".join(sources.values())
    for py, text in sources.items():
        for m in re.finditer(r"_resolve_asset_path\(\s*((?:\"[^\"]*\"\s*,?\s*)+)\)", text):
            cands = ["".join(re.findall(r'"([^"]*)"', s)) for s in re.findall(r'((?:"[^"]*"\s*)+)(?:,|$)', m.group(1))]
            if any(c.startswith(("http", "omniverse")) or (repo / c).exists() or Path(c).exists() for c in cands):
                continue
            bad += 1
            line = text[: m.start()].count("\n") + 1
            owner = (re.findall(r"^class (\w+)", text[: m.start()], re.M) or ["?"])[-1]
            # em uso = montada (X.build/X.attach/X(...)) ou herdada (class Y(X)) em algum módulo da extensão
            body = re.search(rf"^class {owner}\b.*?(?=^class |\Z)", everything, re.M | re.S)
            outside = everything.replace(body.group(0), "") if body else everything
            used = re.search(rf"\b{owner}\.(build|attach)\(|\b{owner}\(|\(\s*{owner}\s*[,)]", outside)
            level, note = ("ERRO", "classe em uso") if used else ("AVISO", "classe sem uso (só importada)")
            add(level, "assets", f"{py.relative_to(repo)}:{line} {owner} ({note}): nenhum candidato existe aqui: {cands}")
    if not bad:
        add("OK", "assets", "todo asset de robô/sensor com caminho no código existe nesta máquina")


def check_paths(repo: Path) -> None:
    files = git(repo, "ls-files", "*.py", "*.sh", "*.json", "*.toml").splitlines()
    rx = re.compile(r"(/home/[A-Za-z0-9_.-]+/[^\s\"'`)]+|/mnt/external/[^\s\"'`)]+|/media/nvme/[^\s\"'`)]+)")
    missing = []
    for rel in files:
        if rel.startswith(("docs/", "legacy_exts/")) or "/tests/" in rel or rel.startswith("tests/") or \
                rel.endswith("check_sim_isolation.py"):
            continue
        try:
            lines = (repo / rel).read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith("#"):
                continue
            for p in rx.findall(line):
                if not os.path.exists(p.rstrip(".,;:")):
                    missing.append(f"{rel}:{i}: {p}")
    for item in missing[:15]:
        add("AVISO", "caminhos", f"caminho de outra máquina, inexistente aqui: {item}")
    if len(missing) > 15:
        add("AVISO", "caminhos", f"... mais {len(missing) - 15}")
    if not missing:
        add("OK", "caminhos", "nenhum caminho absoluto inexistente no código")


def check_worlds(repo: Path) -> None:
    try:
        from pxr import Sdf  # noqa: F401
    except Exception:
        add("AVISO", "mundos", "pxr indisponível neste Python: rode com o env mobgen para checar os mundos")
        return
    from pxr import Sdf
    worlds = sorted((repo / "robot_assets" / "world").glob("*.usd")) + sorted((repo / "robot_assets" / "world").glob("*/*.usd"))
    broken = 0
    for w in worlds:
        if w.name.startswith("_") or "/." in str(w):
            continue
        try:
            layer = Sdf.Layer.FindOrOpen(str(w))
            deps = layer.GetCompositionAssetDependencies() if layer else []
        except Exception as exc:
            add("AVISO", "mundos", f"{w.relative_to(repo)}: não abriu ({exc})")
            continue
        miss = sorted({d for d in deps if not d.startswith(("http", "omniverse")) and not (w.parent / d).exists()
                       and not Path(d).exists()})
        if miss:
            broken += 1
            add("AVISO", "mundos", f"{w.relative_to(repo)}: referências que não resolvem: {miss[:4]}")
    if not broken:
        add("OK", "mundos", f"{len(worlds)} mundos sem referência local quebrada")


def check_sessions(repo: Path, isaac: Path | None, own_ext: str, kind: str, n: int) -> None:
    logs = sorted(glob.glob(str(HOME / ".nvidia-omniverse/logs/Kit/*/*/kit_*.log")), key=os.path.getmtime, reverse=True)
    paths = {str(repo), real(repo)}
    seen = 0
    for log in logs[:200]:
        try:
            with open(log, errors="ignore") as fh:
                head = "".join(next(fh) for _ in range(8))
        except (OSError, StopIteration):
            continue
        cmd = re.search(r"Cmd: (.*)", head)
        if not cmd or not any(p in cmd.group(1) for p in paths):
            continue
        seen += 1
        text = Path(log).read_text(errors="ignore")
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(log)))
        started = re.findall(r"About to startup: \[ext: ([^\]]+)\].*?\(path: ([^)]+)\)", text)
        registered = re.findall(r"\[ext: ([^\]]+)\] registered \(path: ([^)]+)\)", text)
        isaac_root = real(isaac) if isaac else "\0"
        outside = [(e, p) for e, p in registered if not real(p).startswith(isaac_root + "/") and not real(p).startswith(real(repo))]
        own = [p for e, p in started if e.startswith(own_ext)]
        foreign_started = [(e, p) for e, p in started if any(w in (e + p).lower() for w in FOREIGN_WORDS[kind])]
        plugin = "com o plugin" if own else ("SEM o plugin (Isaac puro)" if own_ext not in cmd.group(1) else "plugin NÃO iniciou")
        level = "OK"
        notes = []
        if own and not all(real(p).startswith(real(repo)) for p in own):
            level = "ERRO"
            notes.append(f"{own_ext} carregado de fora do repositório: {own}")
        if foreign_started:
            level = "ERRO"
            notes.append(f"extensões de outro simulador INICIADAS: {foreign_started}")
        if outside:
            notes.append(f"{len(outside)} registradas de fora do Isaac: {', '.join(e for e, _ in outside)[:150]}")
            level = "AVISO" if level == "OK" else level
        if not own and own_ext in cmd.group(1):
            level = "ERRO"
        add(level, "sessoes", f"{when} {Path(log).name}: {plugin}" + ("; " + "; ".join(notes) if notes else ""))
        if seen >= n:
            break
    if not seen:
        add("OK", "sessoes", "nenhuma sessão do Isaac deste repositório nos logs do Kit")


def check_processes(repo: Path) -> None:
    now = time.time()
    boot = float(open("/proc/stat").read().split("btime ")[1].split()[0])
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if not re.search(r"kit/kit |kit/python/bin/python3 ", cmd):
            continue
        age_h = (now - (boot + int(stat[19]) / hz)) / 3600
        ppid = int(stat[1])
        try:
            parent = (Path(f"/proc/{ppid}") / "comm").read_text().strip()
        except OSError:
            parent = "?"
        orphan = ppid == 1 or parent == "systemd"
        level = "AVISO" if orphan and age_h > 1 else "OK"
        add(level, "processos", f"pid {proc.name} há {age_h:.1f} h{' ÓRFÃO' if orphan else ''}: {cmd[:170]}")


# ----------------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--buscar", action="append", default=[], help="outra pasta onde procurar simuladores")
    ap.add_argument("--sessoes", type=int, default=5, help="quantas sessões recentes do Isaac conferir")
    ap.add_argument("--json", default="", help="grava os achados neste JSON")
    ap.add_argument("--quiet", action="store_true", help="só ERRO e AVISO")
    args = ap.parse_args()
    repo = Path(os.path.expanduser(args.repo)).absolute()
    kind = kind_of(repo)
    if not kind:
        print(f"{repo} não tem exts/omni.ext.mobility_gen nem exts/omni.ext.patosim")
        return 1
    own_ext = next(e for e, k in KINDS.items() if k == kind)
    own_modules = toml_modules(repo / "exts" / own_ext / "config" / "extension.toml") or [own_ext]
    repos = find_sim_repos(args.buscar)
    if real(repo) not in map(str, repos):
        repos.append(Path(real(repo)))

    check_repo(repo, kind)
    isaac = check_isaac(repo, repos)
    if isaac:
        check_extsuser(isaac, kind, own_modules)
        check_planner(isaac, repo)
    check_kit_config(kind, own_ext)
    check_conda(repos)
    check_shell(kind)
    check_shortcuts(repo, own_ext)
    check_assets(repo)
    check_paths(repo)
    check_worlds(repo)
    check_sessions(repo, isaac, own_ext, kind, args.sessoes)
    check_processes(repo)

    host = os.uname().nodename
    print(f"# isolamento do {kind} em {host}: {repo}")
    print(f"# simuladores achados: {', '.join(f'{kind_of(r)}={r}' for r in repos)}")
    for f in findings:
        if args.quiet and f["nivel"] == "OK":
            continue
        print(f"[{f['nivel']:5s}] {f['categoria']:10s} {f['msg']}")
    counts = {lvl: sum(f["nivel"] == lvl for f in findings) for lvl in ("ERRO", "AVISO", "OK")}
    print(f"\n{host}: {counts['ERRO']} ERRO, {counts['AVISO']} AVISO, {counts['OK']} OK")
    if args.json:
        Path(args.json).write_text(json.dumps({"host": host, "repo": str(repo), "kind": kind, "contagem": counts,
                                               "achados": findings}, indent=2, ensure_ascii=False))
    return counts["ERRO"]


if __name__ == "__main__":
    raise SystemExit(main())
