import logging
import torch
from pathlib import Path
from torch.optim import lr_scheduler
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import shutil
from timeit import default_timer as timer
import json

from tools.eval_tool import valid, gen_time_str, output_value
from utils.paths import PathManager

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step, warmup_scheduler=None):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }
    if warmup_scheduler is not None:
        save_params["warmup_scheduler"] = warmup_scheduler.state_dict()

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list):
    epoch = config.getint("train", "epoch")
    # batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = Path(config.get("output", "model_path")) / config.get("output", "model_name")
    if output_path.exists():
        logger.warning("Output path exists, check whether need to change a name of model")
    PathManager.ensure_dir(output_path)

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    # Profiling metrics storage
    profiling_metrics = {
        "total_flops": 0,
        "avg_flops_per_batch": 0,
        "profiled_batches": 0
    }

    tensorboard_path = Path(config.get("output", "tensorboard_path")) / config.get("output", "model_name")
    
    if trained_epoch == 0:
        shutil.rmtree(tensorboard_path, ignore_errors=True)

    PathManager.ensure_dir(tensorboard_path)

    writer = SummaryWriter(str(tensorboard_path), config.get("output", "model_name"))

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Scheduler linear com warmup para bert_adam (substitui o StepLR nesse caso)
    warmup_scheduler = None
    optimizer_type = config.get("train", "optimizer")
    if optimizer_type == "bert_adam":
        total_steps = len(dataset) * (epoch - (parameters["trained_epoch"] + 1))
        num_warmup_steps = int(config.getfloat("train", "warmup_ratio") * total_steps)
        warmup_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        if "warmup_scheduler_state" in parameters:
            warmup_scheduler.load_state_dict(parameters["warmup_scheduler_state"])
            logger.info("Warmup scheduler state restored from checkpoint.")
        logger.info(
            "Warmup scheduler criado: total_steps=%d, warmup_steps=%d",
            total_steps, num_warmup_steps
        )

    logger.info("Training start....")

    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")

    total_len = len(dataset)
    if total_len < 10000:
        pass
    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num

        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1
        
        # Profile first 3 batches of first epoch for FLOPs measurement
        should_profile = (current_epoch == trained_epoch)
        
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = data[key].cuda()

            optimizer.zero_grad()

            # Profile specific batches
            if should_profile and step < 3:
                activities = [ProfilerActivity.CPU]
                if len(gpu_list) > 0 and torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)
                
                with profile(
                    activities=activities,
                    record_shapes=True,
                    with_flops=True
                ) as prof:
                    with record_function("model_forward"):
                        results = model(data, config, gpu_list, acc_result, "train")
                        loss, acc_result = results["loss"], results["acc_result"]
                
                # Extract FLOPs from profiler
                total_flops = sum([evt.flops for evt in prof.key_averages() if evt.flops > 0])
                profiling_metrics["total_flops"] += total_flops
                profiling_metrics["profiled_batches"] += 1
                
                logger.info(f"Profiled batch {step}: {total_flops / 1e9:.2f} GFLOPs")
            else:
                results = model(data, config, gpu_list, acc_result, "train")
                loss, acc_result = results["loss"], results["acc_result"]
            
            total_loss += loss.detach().item()

            loss.backward()
            optimizer.step()
            if warmup_scheduler is not None:
                warmup_scheduler.step()

            if step % output_time == 0:
                output_info = output_function(acc_result, config)

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(timer() - start_time), gen_time_str((timer() - start_time) * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

            global_step += 1
            writer.add_scalar(config.get("output", "model_name") + "_train_iter", loss.detach().item(), global_step)
        
        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError
        
        output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(timer() - start_time), gen_time_str((timer() - start_time) * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        checkpoint(str(output_path / f"{current_epoch}.pkl"), model, optimizer, current_epoch, config,
                   global_step, warmup_scheduler=warmup_scheduler)
        writer.add_scalar(config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1),
                          current_epoch)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                eval_res = valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list,
                                 output_function)
                if eval_res is None:
                    pass

        # StepLR só atua quando não há warmup scheduler (outros otimizadores)
        if warmup_scheduler is None:
            exp_lr_scheduler.step()
    
    # Save profiling metrics to file
    if profiling_metrics["profiled_batches"] > 0:
        profiling_metrics["avg_flops_per_batch"] = profiling_metrics["total_flops"] / profiling_metrics["profiled_batches"]
        
        metrics_path = output_path / "profiling_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "total_flops": profiling_metrics["total_flops"],
                "avg_flops_per_batch": profiling_metrics["avg_flops_per_batch"],
                "profiled_batches": profiling_metrics["profiled_batches"],
                "total_gflops": profiling_metrics["total_flops"] / 1e9,
                "avg_gflops_per_batch": profiling_metrics["avg_flops_per_batch"] / 1e9
            }, f, indent=2)
        
        logger.info(f"Profiling complete: {profiling_metrics['avg_flops_per_batch'] / 1e9:.2f} GFLOPs/batch avg")
