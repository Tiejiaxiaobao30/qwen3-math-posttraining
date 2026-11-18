import train_grpo_experiment as exp

def batch_config(world, arm):
    if world != 1 or arm != "baseline":
        raise ValueError("Use one GPU and --arm baseline")
    return dict(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        steps_per_generation=1,
    )

exp.batch_config = batch_config

if __name__ == "__main__":
    exp.main()
