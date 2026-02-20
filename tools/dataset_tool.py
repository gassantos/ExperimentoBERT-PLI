from pathlib import Path
from typing import Union, List


def dfs_search(path: Union[str, Path], recursive: bool) -> List[str]:
    """
    Busca recursiva de arquivos em um diretório.
    
    Args:
        path: Caminho do diretório ou arquivo
        recursive: Se True, busca recursivamente
        
    Returns:
        Lista de caminhos absolutos de arquivos como strings
    """
    path = Path(path)
    
    if path.is_file():
        return [str(path)]
    
    file_list = []
    name_list = sorted(path.iterdir())
    
    for item in name_list:
        if item.is_dir():
            if recursive:
                file_list.extend(dfs_search(item, recursive))
        else:
            file_list.append(str(item))
    
    return file_list
