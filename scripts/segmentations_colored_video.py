import cv2
import numpy as np
from pathlib import Path
import argparse

#vai faltar modificar para as novas cameras e novas pastas de cameras
def create_fixed_colormap():
    """Cria um colormap fixo de 16 cores pastel (em RGB)."""
    colors = [
        [255, 182, 193],  # Light pink
        [176, 224, 230],  # Powder blue
        [255, 218, 185],  # Peach
        [221, 160, 221],  # Plum
        [176, 196, 222],  # Light steel blue
        [152, 251, 152],  # Pale green
        [255, 255, 224],  # Light yellow
        [230, 230, 250],  # Lavender
        [255, 228, 225],  # Misty rose
        [240, 255, 240],  # Honeydew
        [255, 240, 245],  # Lavender blush
        [224, 255, 255],  # Light cyan
        [250, 235, 215],  # Antique white
        [245, 255, 250],  # Mint cream
        [255, 228, 196],  # Bisque
        [240, 248, 255],  # Alice blue
    ]
    return np.array(colors, dtype=np.uint8)


def parse_args():
    """Define e interpreta os argumentos de linha de comando (modular via argparse)."""
    parser = argparse.ArgumentParser(
        description='Coloriza PNGs de segmentação e gera vídeo + frames coloridos.'
    )
    # Mantém os posicionais originais para compatibilidade
    parser.add_argument('input_dir', type=str,
                        help='Diretório com PNGs de segmentação (1 canal).')
    parser.add_argument('output_path', type=str,
                        help='Compat: caminho de vídeo (não é obrigatório ser usado).')

    # Opções
    parser.add_argument('--fps', type=int, default=30, help='Quadros por segundo do vídeo.')
    parser.add_argument('--normals_dir', type=str, help='Diretório com mapas de normais (.npy).')
    parser.add_argument('--depth_dir', type=str,
                        help='Diretório com depth inverso em PNG 16-bit.')
    parser.add_argument('--output_root', type=str, default=None,
                        help='Raiz de saída; pastas serão criadas aqui. '
                             'Se omitido, usa o pai do input_dir.')
    parser.add_argument('--video_name', type=str, default=None,
                        help='Nome do arquivo de vídeo. Default: segmentacao_<ID>.mp4')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='Salvar frames coloridos em disco (default: True).')
    parser.add_argument('--no-save_images', dest='save_images', action='store_false',
                        help='Não salvar frames coloridos (apenas vídeo).')
    parser.add_argument('--save_video', action='store_true', default=True,
                        help='Gerar o vídeo colorido (default: True).')
    parser.add_argument('--no-save_video', dest='save_video', action='store_false',
                        help='Não gerar o vídeo.')
    parser.add_argument('--save_depth_vis', action='store_true', default=False,
                        help='Se fornecido depth_dir, salva visualização 8-bit/colormap.')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Limita o número de frames processados (debug).')

    return parser.parse_args()


def setup_output_dirs(input_dir: Path, output_root: Path):
    """
    Cria (se não existirem) e retorna os diretórios de saída:
    - segmentacao_images_<ID>
    - segmentacao_video_<ID>
    - depth_<ID>
    O <ID> é o nome da pasta de origem (input_dir.name).
    """
    src_id = input_dir.name  # "ID" baseado na pasta de origem
    images_dir = output_root / f"segmentacao_images_{src_id}"
    video_dir = output_root / f"segmentacao_video_{src_id}"
    depth_vis_dir = output_root / f"depth_{src_id}"

    # Cria apenas se não existir (exist_ok=True evita erro e não “recria”)
    images_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_dir.mkdir(parents=True, exist_ok=True)

    return src_id, images_dir, video_dir, depth_vis_dir


def to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """Converte RGB para BGR (OpenCV trabalha em BGR)."""
    return img_rgb[..., ::-1]


def visualize_depth_16u(depth_16u: np.ndarray) -> np.ndarray:
    """
    Gera uma visualização 8-bit colorizada a partir de um depth inverso 16-bit.
    Normaliza no intervalo [0,255] e aplica um colormap.
    """
    # Normaliza (protege contra constantes)
    dmin, dmax = int(depth_16u.min()), int(depth_16u.max())
    if dmax == dmin:
        vis8 = np.zeros_like(depth_16u, dtype=np.uint8)
    else:
        vis8 = cv2.normalize(depth_16u, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Colormap (BGR)
    vis_color = cv2.applyColorMap(vis8, cv2.COLORMAP_JET)
    return vis_color


def main():
    args = parse_args()

    # --- Descoberta de paths de entrada ---
    input_dir = Path(args.input_dir)
    png_files = sorted(list(input_dir.glob('*.png')))
    if not png_files:
        raise ValueError(f"Nenhum PNG encontrado em {input_dir}")

    normals_dir = Path(args.normals_dir) if args.normals_dir else None
    depth_dir = Path(args.depth_dir) if args.depth_dir else None

    # --- Diretórios de saída e ID da origem ---
    output_root = Path(args.output_root) if args.output_root else input_dir.parent
    src_id, images_dir, video_dir, depth_vis_dir = setup_output_dirs(input_dir, output_root)

    # --- Lê o primeiro frame para obter dimensões ---
    first_img = cv2.imread(str(png_files[0]), cv2.IMREAD_UNCHANGED)
    if first_img is None:
        raise RuntimeError(f"Falha ao ler {png_files[0]}")
    if first_img.ndim == 2:
        height, width = first_img.shape
    else:
        height, width = first_img.shape[:2]

    # --- Colormap fixo ---
    colormap = create_fixed_colormap()
    n_colors = len(colormap)

    # --- Preparação do vídeo (se habilitado) ---
    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = args.video_name if args.video_name else f"segmentacao_{src_id}.mp4"
        # Se o usuário passou um output_path “antigo”, use esse nome, mas salve na pasta de vídeo
        if args.output_path and args.video_name is None:
            video_name = Path(args.output_path).name
        video_path = video_dir / video_name
        out = cv2.VideoWriter(str(video_path), fourcc, args.fps, (width, height))

    # --- Loop de processamento ---
    processed = 0
    for png_file in png_files:
        # Limite de frames (opcional)
        if args.max_frames is not None and processed >= args.max_frames:
            break

        # 1) Carrega a segmentação (1 canal, índices de classe)
        seg = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
        if seg is None:
            print(f"[AVISO] Falha ao ler {png_file}, pulando…")
            continue
        if seg.ndim > 2:
            # Se vier com canais, pega o primeiro (garantia)
            seg = seg[..., 0]
        # Garanta tipo inteiro para indexar colormap
        seg_idx = seg.astype(np.int64)

        # 2) (Opcional) Carrega normal map (formato .npy, com 4 canais esperados)
        normal_map = None
        if normals_dir:
            normal_file = normals_dir / (png_file.stem + '.npy')
            if normal_file.exists():
                try:
                    normal_map = np.load(str(normal_file))
                except Exception as e:
                    print(f"[AVISO] Não foi possível carregar {normal_file}: {e}")

        # 3) (Opcional) Carrega depth 16-bit inverso
        depth_map_16u = None
        if depth_dir:
            depth_file = depth_dir / png_file.name
            if depth_file.exists():
                depth_map_16u = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                if depth_map_16u is None:
                    print(f"[AVISO] Não foi possível ler {depth_file}")
            else:
                print(f"[AVISO] Depth não encontrado: {depth_file}")

        # 4) Coloriza a segmentação (RGB) e aplica sombreamento (normais/depth)
        colored_frame_rgb = colormap[seg_idx % n_colors]

        if normal_map is not None or depth_map_16u is not None:
            # Sombreamento multiplicativo em [0,1]
            shading = np.ones((height, width), dtype=np.float32)

            if normal_map is not None:
                # Converte normais de [0,1] para [-1,1]
                normal_xyz = 2.0 * normal_map[..., :3].astype(np.float32) - 1.0
                # Luz direcional simples
                light_dir = np.array([-0.5, 0.5, 1.0], dtype=np.float32)
                light_dir /= np.linalg.norm(light_dir) + 1e-8
                # Lambertiano: max(dot(n,l), 0)
                # (broadcast implícito por reshape)
                nl = (normal_xyz * light_dir.reshape(1, 1, 3)).sum(axis=2)
                nl = np.clip(nl, 0.0, 1.0)

                # Se houver canal 4 (alpha/força), normaliza de forma estável
                if normal_map.shape[-1] >= 4:
                    denom = normal_map[..., 3].astype(np.float32)
                    denom = np.where(denom == 0, 1.0, denom)
                    nl = nl / denom

                # Eleva contraste leve
                normal_shading = 0.5 + 0.5 * nl
                shading *= normal_shading

            if depth_map_16u is not None:
                # Exemplo simples: um leve ganho com base no depth normalizado
                # (comenta “pass” anterior; ajuste ao seu gosto)
                depth_norm = cv2.normalize(depth_map_16u.astype(np.float32),
                                           None, 0.0, 1.0, cv2.NORM_MINMAX)
                # Exponenciação suave
                shading *= (0.9 + 0.1 * depth_norm**0.3)

            # Aplica no frame (convertendo para float antes)
            colored_frame_rgb = (colored_frame_rgb.astype(np.float32) *
                                 shading[..., None]).clip(0, 255).astype(np.uint8)

        # OpenCV espera BGR
        colored_frame_bgr = to_bgr(colored_frame_rgb)

        # 5) Salva frame colorido em imagem (se habilitado)
        if args.save_images:
            out_img_path = images_dir / f"{png_file.stem}.png"
            cv2.imwrite(str(out_img_path), colored_frame_bgr)

        # 6) Escreve no vídeo (se habilitado)
        if out is not None:
            out.write(colored_frame_bgr)

        # 7) (Opcional) salva visualização de depth
        if args.save_depth_vis and depth_map_16u is not None:
            depth_vis = visualize_depth_16u(depth_map_16u)
            cv2.imwrite(str(depth_vis_dir / f"{png_file.stem}.png"), depth_vis)

        processed += 1

    # Libera o writer de vídeo
    if out is not None:
        out.release()
        print(f"[OK] Vídeo salvo em: {video_dir}")

    if args.save_images:
        print(f"[OK] Frames coloridos salvos em: {images_dir}")

    if args.save_depth_vis and depth_dir is not None:
        print(f"[OK] Visualizações de profundidade salvas em: {depth_vis_dir}")

    print(f"[DONE] Processados {processed} frame(s) de {len(png_files)}.")


if __name__ == "__main__":
    main()
