"""
Módulo para gerenciamento de caminhos multiplataforma.
Usa pathlib para garantir compatibilidade entre sistemas operacionais.
"""
from pathlib import Path
from typing import Union, Optional
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração do cache do HuggingFace para o diretório LOCAL do projeto.
# DEVE ser executado ANTES de qualquer import de transformers/huggingface_hub,
# por isso está no nível de módulo de utils/paths (importado primeiro).
# Usa setdefault para respeitar sobrescritas via variável de ambiente externa.
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent
_HF_PRETRAINED_DIR = _BASE_DIR / "examples" / "pretrain_models"
_HF_PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(_HF_PRETRAINED_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_PRETRAINED_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_PRETRAINED_DIR / "hub"))


class PathManager:
    """Gerenciador de caminhos do projeto com suporte multiplataforma."""
    
    # Diretório base do projeto
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Diretórios principais
    CACHE_DIR = BASE_DIR / ".cache"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    RESULTS_DIR = BASE_DIR / "results"
    OUTPUT_DIR = BASE_DIR / "output"
    EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
    CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

    # Diretório local para artefatos de modelos pré-treinados (HuggingFace cache)
    HF_PRETRAINED_DIR = BASE_DIR / "examples" / "pretrain_models"
    HF_HUB_CACHE_DIR = HF_PRETRAINED_DIR / "hub"
    
    
    @classmethod
    def setup_directories(cls, verbose: bool = True):
        """
        Cria todos os diretórios necessários se não existirem.
        
        Args:
            verbose: Se True, loga criação de diretórios
        """
        directories = [
            cls.CACHE_DIR,
            cls.DATA_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR,
            cls.OUTPUT_DIR,
            cls.EXPERIMENTS_DIR,
            cls.CHECKPOINTS_DIR,
        ]
        
        for dir_path in directories:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                if verbose:
                    logger.info(f"Created directory: {dir_path}")
            elif verbose:
                logger.debug(f"Directory already exists: {dir_path}")
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Garante que o diretório existe, criando se necessário.
        
        Args:
            path: Caminho do diretório
        
        Returns:
            Path: Objeto Path do diretório
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def safe_path(path: Union[str, Path]) -> Path:
        """
        Converte string para Path de forma segura, expandindo variáveis.
        
        Args:
            path: Caminho como string ou Path
        
        Returns:
            Path: Objeto Path normalizado
        """
        if isinstance(path, str):
            # Expande ~ para home directory
            path = os.path.expanduser(path)
            # Expande variáveis de ambiente
            path = os.path.expandvars(path)
        
        return Path(path).resolve()
    
    @staticmethod
    def get_relative_path(path: Union[str, Path], base: Optional[Union[str, Path]] = None) -> Path:
        """
        Retorna caminho relativo a partir de uma base.
        
        Args:
            path: Caminho absoluto ou relativo
            base: Diretório base (padrão: BASE_DIR)
        
        Returns:
            Path: Caminho relativo
        """
        path = Path(path).resolve()
        base = Path(base).resolve() if base else PathManager.BASE_DIR
        
        try:
            return path.relative_to(base)
        except ValueError:
            # Se não for possível calcular relativo, retorna absoluto
            logger.warning(f"Cannot compute relative path for {path} from {base}")
            return path
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], must_exist: bool = True, 
                          extensions: Optional[list] = None) -> Path:
        """
        Valida um caminho de arquivo.
        
        Args:
            path: Caminho do arquivo
            must_exist: Se True, arquivo deve existir
            extensions: Lista de extensões válidas (ex: ['.csv', '.json'])
        
        Returns:
            Path: Path validado
        
        Raises:
            FileNotFoundError: Se arquivo não existe e must_exist=True
            ValueError: Se extensão inválida
        """
        path = PathManager.safe_path(path)
        
        if must_exist and not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if must_exist and not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        if extensions:
            if path.suffix.lower() not in [ext.lower() for ext in extensions]:
                raise ValueError(
                    f"Invalid file extension: {path.suffix}. "
                    f"Expected one of: {extensions}"
                )
        
        return path
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Retorna caminho completo para arquivo de dados."""
        return cls.DATA_DIR / filename
       
    @classmethod
    def get_log_path(cls, log_name: str) -> Path:
        """Retorna caminho completo para arquivo de log."""
        return cls.LOGS_DIR / log_name
    
    @classmethod
    def get_output_path(cls, output_name: str) -> Path:
        """Retorna caminho completo para arquivo de output."""
        return cls.OUTPUT_DIR / output_name

    @classmethod
    def get_experiment_path(cls, experiment_name: str) -> Path:
        """Retorna caminho completo para diretório de experimento."""
        return cls.EXPERIMENTS_DIR / experiment_name
    
    @classmethod
    def get_checkpoint_path(cls, checkpoint_name: str) -> Path:
        """Retorna caminho completo para checkpoint."""
        return cls.CHECKPOINTS_DIR / checkpoint_name
    
    @classmethod
    def get_results_path(cls, results_name: str) -> Path:
        """Retorna caminho completo para arquivo de resultados."""
        return cls.RESULTS_DIR / results_name
    
    @classmethod
    def get_projectdir(cls) -> Path:
        """Retorna caminho completo para diretório do projeto."""
        return cls.BASE_DIR


# Inicializa diretórios ao importar o módulo
try:
    PathManager.setup_directories(verbose=False)
except Exception as e:
    logger.warning(f"Could not create all directories: {e}")