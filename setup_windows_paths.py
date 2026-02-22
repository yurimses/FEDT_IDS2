#!/usr/bin/env python3
"""
Script auxiliar para converter caminhos Linux para Windows nos scripts de gráficos.

USO:
    python setup_windows_paths.py C:/Users/SeuUsuario/FEDT_IDS2

ATENÇÃO: Este script irá modificar os arquivos. Faça backup antes!
"""

import sys
from pathlib import Path
import re


def convert_paths_in_file(filepath: Path, old_base: str, new_base: str, dry_run=False):
    """
    Substitui caminhos Linux por Windows em um arquivo Python.
    
    Args:
        filepath: Caminho do arquivo a modificar
        old_base: Caminho base Linux (ex: /home/yuri/FEDT_IDS2)
        new_base: Caminho base Windows (ex: C:/Users/SeuUsuario/FEDT_IDS2)
        dry_run: Se True, apenas mostra o que seria modificado sem alterar
    """
    if not filepath.exists():
        print(f"⚠️  Arquivo não encontrado: {filepath}")
        return False
    
    # Lê o conteúdo do arquivo
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Conta quantas substituições serão feitas
    count = content.count(old_base)
    
    if count == 0:
        print(f"✓  {filepath.name} - Nenhuma alteração necessária")
        return False
    
    # Faz a substituição
    new_content = content.replace(old_base, new_base)
    
    if dry_run:
        print(f"🔍 {filepath.name} - {count} substituições encontradas (DRY RUN)")
        return True
    
    # Salva o arquivo modificado
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ {filepath.name} - {count} caminhos atualizados")
    return True


def main():
    # Configuração
    OLD_BASE_PATH = "/home/yuri/FEDT_IDS2"
    
    if len(sys.argv) < 2:
        print("❌ Erro: Caminho do Windows não fornecido!")
        print()
        print("USO:")
        print("    python setup_windows_paths.py C:/Users/SeuUsuario/FEDT_IDS2")
        print()
        print("ATENÇÃO: Use / (barra normal) no caminho, NÃO use \\")
        print()
        print("Exemplo:")
        print('    python setup_windows_paths.py "C:/Users/Maria/Documents/FEDT_IDS2"')
        sys.exit(1)
    
    NEW_BASE_PATH = sys.argv[1].replace("\\", "/")  # Normaliza barras
    
    # Verifica se o caminho está correto
    if not Path(NEW_BASE_PATH).exists():
        print(f"⚠️  AVISO: O caminho {NEW_BASE_PATH} não existe!")
        response = input("Deseja continuar mesmo assim? (s/N): ")
        if response.lower() not in ['s', 'sim', 'y', 'yes']:
            print("Operação cancelada.")
            sys.exit(0)
    
    print("=" * 70)
    print("CONVERSÃO DE CAMINHOS LINUX → WINDOWS")
    print("=" * 70)
    print(f"Caminho Linux:   {OLD_BASE_PATH}")
    print(f"Caminho Windows: {NEW_BASE_PATH}")
    print()
    
    # Lista de arquivos a modificar
    project_root = Path(__file__).parent
    
    files_to_convert = [
        project_root / "scripts" / "graficos_unlearning.py",
        project_root / "scripts" / "graficos_edgeiot.py",
        project_root / "scripts" / "graficos_edgeiot_20rounds.py",
        project_root / "scripts" / "comparison_fedt_baselines.py",
        project_root / "scripts" / "bkp" / "graficos_edgeiot.py",
    ]
    
    # Primeiro, faz um dry run para mostrar o que será modificado
    print("🔍 Verificando arquivos...")
    print()
    
    changes_needed = []
    for filepath in files_to_convert:
        if convert_paths_in_file(filepath, OLD_BASE_PATH, NEW_BASE_PATH, dry_run=True):
            changes_needed.append(filepath)
    
    if not changes_needed:
        print()
        print("✓ Nenhum arquivo precisa ser modificado!")
        sys.exit(0)
    
    # Pede confirmação
    print()
    print(f"📝 {len(changes_needed)} arquivo(s) serão modificados:")
    for fp in changes_needed:
        print(f"   - {fp.relative_to(project_root)}")
    
    print()
    response = input("Deseja prosseguir com as modificações? (s/N): ")
    
    if response.lower() not in ['s', 'sim', 'y', 'yes']:
        print("Operação cancelada.")
        sys.exit(0)
    
    # Faz as modificações
    print()
    print("✏️  Modificando arquivos...")
    print()
    
    for filepath in changes_needed:
        convert_paths_in_file(filepath, OLD_BASE_PATH, NEW_BASE_PATH, dry_run=False)
    
    print()
    print("=" * 70)
    print("✅ CONVERSÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    print()
    print("Próximos passos:")
    print("1. Verifique se os caminhos estão corretos nos arquivos modificados")
    print("2. Teste os scripts:")
    print(f"   python scripts\\graficos_unlearning.py")
    print()


if __name__ == "__main__":
    main()
