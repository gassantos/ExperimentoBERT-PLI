import platform
import psutil
import torch
from datetime import datetime

def get_cpu_info():
    """Coleta informações sobre a CPU."""
    return {
        'processor': platform.processor(),
        'physical_cores': psutil.cpu_count(logical=False),
        'total_cores': psutil.cpu_count(logical=True),
        'max_frequency': f"{psutil.cpu_freq().max:.2f} Mhz" if psutil.cpu_freq() else "N/A",
        'current_frequency': f"{psutil.cpu_freq().current:.2f} Mhz" if psutil.cpu_freq() else "N/A",
        'cpu_usage_percent': psutil.cpu_percent(interval=1)
    }


def get_ram_info():
    """Coleta informações sobre a memória RAM."""
    ram = psutil.virtual_memory()
    return {
        'total': f"{ram.total / (1024**3):.2f} GB",
        'available': f"{ram.available / (1024**3):.2f} GB",
        'used': f"{ram.used / (1024**3):.2f} GB",
        'percent_used': f"{ram.percent}%"
    }


def get_gpu_info():
    """Coleta informações sobre a GPU."""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpus': []
    }
    
    if torch.cuda.is_available():
        gpu_info['gpu_count'] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            gpu_info['gpus'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB",
                'memory_allocated': f"{torch.cuda.memory_allocated(i) / (1024**3):.2f} GB",
                'memory_reserved': f"{torch.cuda.memory_reserved(i) / (1024**3):.2f} GB"
            })
    
    return gpu_info


def get_disk_info():
    """Coleta informações sobre o disco."""
    partitions = psutil.disk_partitions()
    disk_info = []
    
    for partition in partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_info.append({
                'device': partition.device,
                'mountpoint': partition.mountpoint,
                'fstype': partition.fstype,
                'total': f"{usage.total / (1024**3):.2f} GB",
                'used': f"{usage.used / (1024**3):.2f} GB",
                'free': f"{usage.free / (1024**3):.2f} GB",
                'percent_used': f"{usage.percent}%"
            })
        except PermissionError:
            continue
    
    return disk_info


def get_device_info():
    """Coleta informações sobre o dispositivo."""
    return {
        'system': platform.system(),
        'node_name': platform.node(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'architecture': platform.architecture()[0]
    }


def get_torch_device():
    """Retorna o dispositivo PyTorch disponível."""
    if torch.cuda.is_available():
        return {
            'type': 'cuda',
            'name': torch.cuda.get_device_name(0),
            'device': torch.device('cuda')
        }
    else:
        return {
            'type': 'cpu',
            'name': platform.processor(),
            'device': torch.device('cpu')
        }


def collect_system_info():
    """Coleta todas as informações do sistema de forma dinâmica."""
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cpu': get_cpu_info(),
        'ram': get_ram_info(),
        'gpu': get_gpu_info(),
        'disk': get_disk_info(),
        'device': get_device_info(),
        'torch_device': get_torch_device()
    }


def print_system_info():
    """Imprime as informações do sistema de forma formatada."""
    info = collect_system_info()
    
    print("=" * 50)
    print(f"System Information - {info['timestamp']}")
    print("=" * 50)
    
    print("\n[CPU Information]")
    for key, value in info['cpu'].items():
        print(f"  {key}: {value}")
    
    print("\n[RAM Information]")
    for key, value in info['ram'].items():
        print(f"  {key}: {value}")
    
    print("\n[GPU Information]")
    print(f"  CUDA Available: {info['gpu']['cuda_available']}")
    print(f"  GPU Count: {info['gpu']['gpu_count']}")
    for gpu in info['gpu']['gpus']:
        print(f"\n  GPU {gpu['id']}:")
        for key, value in gpu.items():
            if key != 'id':
                print(f"    {key}: {value}")
    
    print("\n[Device Information]")
    for key, value in info['device'].items():
        print(f"  {key}: {value}")
    
    print("\n[PyTorch Device]")
    for key, value in info['torch_device'].items():
        if key != 'device':
            print(f"  {key}: {value}")
    
    print("=" * 50)